import os
import pandas as pd
import tempfile
import io
import re
from collections import Counter
from flask import Flask, request, render_template, session, redirect, send_file, after_this_request, jsonify
from dotenv import load_dotenv
from rapidfuzz import process as fuzzy_process, fuzz
from country_mapping import load_country_data
from flask_session import Session
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
#---------------------------------------------------------------------------------------------------------------------

def robust_detect_header_row(file_path, file_type="excel", sheet_name=None, max_skip=20, debug=False):
    """
    Try using each of the first `max_skip` rows as headers.
    Returns the earliest row with the widest usable column structure.
    """
    shapes = []

    for skip in range(max_skip):
        try:
            if file_type == "excel":
                df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip)
            elif file_type == "csv":
                df = pd.read_csv(file_path, skiprows=skip)
            else:
                raise ValueError("Unsupported file type.")

            df = df.dropna(how='all')  # drop blank rows
            n_cols = df.shape[1]
            if debug:
                print(f"TRYING skiprows={skip}: shape={df.shape}")
            shapes.append((skip, n_cols))
        except Exception as e:
            if debug:
                print(f"Failed skiprows={skip}: {e}")
            continue

    if not shapes:
        raise ValueError(f"No usable structure found in first {max_skip} rows of {file_type}.")

    # Find the maximum width observed
    max_width = max(width for _, width in shapes)
    # Return the first skip that gives max_width
    for skip, width in shapes:
        if width == max_width:
            try:
                if debug:
                    print(f"✅ USING skiprows={skip} with width={width}")
                if file_type == "excel":
                    return pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip)
                else:
                    return pd.read_csv(file_path, skiprows=skip)
            except Exception as e:
                if debug:
                    print(f"Failed final read at skiprows={skip}: {e}")
                continue

    raise ValueError("Could not load a stable header row.")


def robust_read_csv(file_path):
    return robust_detect_header_row(file_path, file_type="csv")


def robust_read_excel(file_path, sheet_name=None):
    return robust_detect_header_row(file_path, file_type="excel", sheet_name=sheet_name)


def is_non_numeric_string(x):
    if not isinstance(x, str):
        return False
    s = x.strip()
    if not s:
        return False
    # Case 1: Pure number (e.g., "2020", "-1.5", ".99", "1e6")
    if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", s):
        return False
    # Case 2: Number with units/symbols
    has_number = re.search(r"\d", s)
    has_unit = re.search(r"(km|kg|miles|years|°c|°f|usd|eur|gbp|¥|€|\$|%|lbs|tons)", s, flags=re.IGNORECASE)
    if has_number and has_unit:
        return False
    return True


def extract_match_candidates(df):
    flat_values = pd.Series(df.values.ravel())
    header_values = pd.Series(df.columns.astype(str))
    index_values = pd.Series(df.index.astype(str))

    combined = pd.concat([flat_values, header_values, index_values])

    unique_strings = combined[combined.apply(is_non_numeric_string)].unique()
    return [x.strip() for x in unique_strings]
#---------------------------------------------------------------------------------------------------------------------

# Load environment variables
load_dotenv()
# App setup
app = Flask(__name__, template_folder='html')
# This tells Flask to store session data on disk in a temporary directory, which allows much larger payloads
app.config['SESSION_TYPE'] = 'filesystem'  # Store sessions in the local file system
app.config['SESSION_PERMANENT'] = False
# Enforce a size limit of 30mb to the app
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30 MB
Session(app)
#
app.secret_key = os.getenv('SECRET_KEY')
#---------------------------------------------------------------------------------------------------------------------

@app.route('/')
def upload():
    """
    Shows upload form.
    If file already uploaded (stored in session), displays file name.
    If Excel, shows sheet selection.
    If sheet selected, shows data preview.
    """
    file_name = session.get('file_name')
    sheet_names = session.get('sheet_names')  # set earlier if Excel
    return render_template("upload.html", file_name=file_name, sheet_names=sheet_names)


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Saves uploaded file and shows sheet picker if Excel.
    For CSV, disables sheet dropdown but keeps UI consistent.
    """
    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('upload.html', error="No file selected.")

    filename = file.filename
    temp_path = tempfile.mktemp(suffix=os.path.splitext(filename)[-1])
    file.save(temp_path)

    session['temp_file_path'] = temp_path
    session['file_name'] = filename

    if filename.endswith('.xlsx'):
        try:
            xls = pd.ExcelFile(temp_path)
            session['sheets'] = xls.sheet_names
            return render_template(
                'upload.html',
                file_name=filename,
                sheets=xls.sheet_names
            )
        except Exception as e:
            return render_template('upload.html', error=f"Error reading Excel file: {e}")

    elif filename.endswith('.csv'):
        try:
            _ = robust_read_csv(temp_path)  # test parsing only
            session['sheets'] = ["(CSV files have no sheets)"]
            session['sheet_name'] = None
            return render_template(
                'upload.html',
                file_name=filename,
                sheets=session['sheets']
            )
        except Exception as e:
            return render_template(
                'upload.html',
                error=f"Error reading CSV file: {e}",
                file_name=filename,
                sheets=[]
            )

    else:
        return render_template('upload.html', error="Unsupported file format.")


@app.route('/preview_data', methods=['GET','POST'])
def preview_data():
    """
    Reads the uploaded file (CSV or Excel + sheet).
    Stores a preview (max 20 rows × 30 columns).
    Masks middle columns if >30.
    Renders the upload.html template with preview + sheet info.    
    """
    file_path = session.get('temp_file_path')
    file_name = session.get('file_name')
    sheet_name = request.form.get('sheet_name') if request.method == 'POST' else session.get('sheet_name')
    selected_rows = session.get('selected_rows', [])
    selected_cols = session.get('selected_cols', [])
    print("🧠 selected_rows in session:", session.get('selected_rows'))
    print("🧠 selected_cols in session:", session.get('selected_cols'))



    if not file_path or not file_name:
        return redirect('/')

    try:
        # Read the file based on extension using robust logic
        if file_name.endswith('.xlsx'):
            session['sheet_name'] = sheet_name
            df = robust_read_excel(file_path, sheet_name=sheet_name)

        elif file_name.endswith('.csv'):
            df = robust_read_csv(file_path)
            session['sheet_name'] = None

        else:
            return render_template("upload.html", error="Unsupported file type.", file_name=file_name, sheets=session.get('sheets'))

    except Exception as e:
        return render_template("upload.html", error=f"Error reading file: {e}", file_name=file_name, sheets=session.get('sheets'))

    # Convert to string and fill NA
    df = df.fillna('').astype(str)

    # Limit preview to 20 rows
    df_preview = df.head(20)
    #print("PREVIEW DF SHAPE:", df.shape)

    # Limit to 30 columns, insert "..." column if needed
    if df_preview.shape[1] > 30:
        first_cols = df_preview.iloc[:, :15]
        last_cols = df_preview.iloc[:, -15:]
        masked_col = pd.DataFrame({"...": ["..."] * df_preview.shape[0]})
        df_preview = pd.concat([first_cols, masked_col, last_cols], axis=1)

    # Prepare data for rendering
    preview_table_data = [
        {"index": idx, "row": list(row)}
        for idx, row in zip(df_preview.index, df_preview.itertuples(index=False, name=None))
    ]

    session['preview_columns'] = df_preview.columns.tolist()
    session['preview_data'] = preview_table_data
    session['num_rows'] = df.shape[0]

    return render_template(
        'upload.html',
        file_name=file_name,
        sheet_name=sheet_name,
        sheets=session.get('sheets'),
        columns=df_preview.columns.tolist(),  
        preview_data=preview_table_data,
        num_rows=df.shape[0],
        selected_rows=selected_rows,
        selected_cols=selected_cols,
        threshold=session.get('threshold', 80)
    )


@app.route('/process_selection', methods=['GET','POST'])
def process_selection():
    file_path = session.get('temp_file_path')
    sheet_name = session.get('sheet_name')
    file_name = session.get('file_name')
    threshold = int(request.form.get("threshold", 80))
    session["threshold"] = threshold

    if not file_path or not file_name:
        return redirect('/')

    try:
        if file_name.endswith('.csv'):
            df = robust_read_csv(file_path)
        else:
            df = robust_read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        return render_template("upload.html", error=f"Error reading data: {e}", file_name=file_name, sheets=session.get('sheets'))

    df = df.fillna('').astype(str)

    # Get and store raw selections
    row_ids = session.get('selected_rows', [])
    col_ids = session.get('selected_cols', [])
    # session['selected_rows'] = row_ids
    # session['selected_cols'] = col_ids

    # Parse row/column selections
    row_indices = [int(r.split('-')[1]) for r in row_ids if r.startswith('row-')]
    col_indices = [int(c.split('-')[1]) for c in col_ids if c.startswith('col-')]
    include_index = 'index' in row_ids
    include_header = 'header' in col_ids

    # Start with empty DataFrame and build explicitly
    if col_indices:
        selected_df = df.iloc[:, col_indices]
    else:
        selected_df = df.copy()

    if row_indices:
        selected_df = selected_df.iloc[row_indices, :]
    # else: keep all rows

    # Optional debug
    print("🔎 Selected shape:", selected_df.shape)
    print("🔎 Selected rows:", row_indices if row_indices else "ALL")
    print("🔎 Selected cols:", col_indices if col_indices else "ALL")

    # Build only from selected cells
    flat_values = pd.Series(selected_df.values.ravel())

    # Add optional header/index values
    index_strings = df.index[row_indices].astype(str) if include_index and row_indices else (
        df.index.astype(str) if include_index else pd.Index([])
    )
    header_strings = df.columns[col_indices].astype(str) if include_header and col_indices else (
        df.columns.astype(str) if include_header else pd.Index([])
    )

    combined = pd.concat([flat_values, pd.Series(index_strings), pd.Series(header_strings)])
    unique_strings = combined[combined.apply(is_non_numeric_string)].unique()


    # --- Country Matching ---
    country_docs = load_country_data()
    name_to_standard = {}
    for doc in country_docs:
        std = doc["ifs_name"]
        fipscode = doc["ifs_fipscode"]
        name_to_standard[std.lower()] = std
        name_to_standard[fipscode.lower()] = std
        for alt in doc.get("alternative_names", []):
            name_to_standard[alt.lower()] = std

    matchable_names = list(name_to_standard.keys())

    unmatched_countries = {}
    for original in unique_strings:
        matches = fuzzy_process.extract(original.lower(), matchable_names, scorer=fuzz.token_sort_ratio, limit=None)
        deduped = {}
        for alt_name, score, _ in matches:
            std_name = name_to_standard[alt_name]
            if std_name not in deduped or score > deduped[std_name]:
                deduped[std_name] = score

        suggestions = [(name, round(score)) for name, score in deduped.items()]
        unmatched_countries[original] = sorted(suggestions, key=lambda x: -x[1])

    confirmed_mappings = {
        k: v[0][0] for k, v in unmatched_countries.items() if v and v[0][1] >= threshold
    }

    session['unmatched_countries'] = unmatched_countries
    session['confirmed_mappings'] = confirmed_mappings

    return redirect('/review_matches?page=1')


@app.route('/save_selection', methods=['POST'])
def save_selection():
    data = request.json
    session['selected_rows'] = data.get('rows', [])
    session['selected_cols'] = data.get('cols', [])

    # Force session to persist before response ends
    session.modified = True

    return jsonify(success=True)


@app.route('/review_matches', methods=['GET', 'POST'])
def review_matches():
    unmatched = session.get('unmatched_countries', {})
    confirmed = session.get('confirmed_mappings', {})

    if request.method == 'POST':
        action = request.form.get('action', '')
        per_page = int(request.form.get('per_page', 200))
        current_page = int(request.form.get('current_page', 1))

        original_names = request.form.getlist('original_names[]')
        standard_names = request.form.getlist('standard_names[]')

        if action not in {'reset_page', 'reset_all', 'finish'}:
            if original_names and standard_names:
                page_mapping = dict(zip(original_names, standard_names))
                new_confirmed = dict(confirmed)
                new_confirmed.update(page_mapping)
                session['confirmed_mappings'] = new_confirmed

        # # 🔁 Re-fetch clean confirmed after any overwrite
        # confirmed = session.get('confirmed_mappings', {}) 
        # Determine action
        if action == 'next':
            current_page += 1
        elif action == 'prev':
            current_page = max(1, current_page - 1)
        elif action.startswith("go_to_"):
            try:
                current_page = int(action.split("_")[-1])
            except ValueError:
                current_page = 1
        elif action == 'change_per_page':
            current_page = 1
            per_page = int(request.form.get('per_page', 200))
        # Reset logic now uses fresh version
        elif action == 'reset_page':
            new_confirmed = dict(confirmed)
            for name in original_names:
                default_suggestion = unmatched.get(name, [])
                new_confirmed[name] = default_suggestion[0][0] if default_suggestion else ""
            session['confirmed_mappings'] = new_confirmed
            return redirect(f"/review_matches?page={current_page}&per_page={per_page}")
        elif action == 'reset_all':
            return redirect('/reset_matches')
        elif action == 'save_and_go':
            return redirect(f"/review_matches?page={current_page}&per_page={per_page}")
        elif action == 'finish':
            original_names = request.form.getlist('original_names[]')
            standard_names = request.form.getlist('standard_names[]')

            if original_names and standard_names:
                page_mapping = dict(zip(original_names, standard_names))
                new_confirmed = dict(confirmed)
                new_confirmed.update(page_mapping)
                session['confirmed_mappings'] = new_confirmed
            return redirect('/download_file')
        return redirect(f"/review_matches?page={current_page}&per_page={per_page}")

    # GET request: show current page
    per_page = int(request.args.get('per_page', 200))
    current_page = int(request.args.get('page', 1))
    unmatched = session.get('unmatched_countries', {})
    confirmed = session.get('confirmed_mappings', {})

    total_entries = len(unmatched)
    total_pages = max(1, (total_entries + per_page - 1) // per_page)
    keys = list(unmatched.keys())
    page_keys = keys[(current_page - 1) * per_page: current_page * per_page]

    page_data = {k: unmatched[k] for k in page_keys}
    default_selections = {k: confirmed.get(k, "") for k in page_keys}

    country_docs = load_country_data()
    standard_to_fipscode = {
        doc['ifs_name']: doc['ifs_fipscode']
        for doc in country_docs if 'ifs_name' in doc and 'ifs_fipscode' in doc
    }

    all_standard_names = sorted({
        doc["ifs_name"] for doc in country_docs if "ifs_name" in doc
    })

    return render_template(
        "edit_table.html",
        unmatched_countries=page_data,
        default_selections=default_selections,
        current_page=current_page,
        total_pages=total_pages,
        per_page=per_page,
        total_entries=total_entries,
        fipscodes=standard_to_fipscode,
        all_standard_names=all_standard_names
    )


@app.route('/reset_matches')
def reset_matches():
    unmatched = session.get('unmatched_countries', {})
    threshold = session.get("threshold", 80)
    scope = request.args.get('scope', 'all')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 200))
    keys = list(unmatched.keys())

    if scope == 'page':
        page_keys = keys[(page - 1) * per_page: page * per_page]
        confirmed = session.get('confirmed_mappings', {})
        for name in page_keys:
            suggestion = unmatched.get(name, [])
            confirmed[name] = suggestion[0][0] if suggestion and suggestion[0][1] >= threshold else ""
        session['confirmed_mappings'] = confirmed
    else:
        confirmed = {
            k: v[0][0] for k, v in unmatched.items()
            if v and v[0][1] >= threshold
        }
        session['confirmed_mappings'] = confirmed

    return redirect(f'/review_matches?page={page}&per_page={per_page}')


@app.route('/download_file')
def download_file():
    confirmed = session.get('confirmed_mappings', {})
    file_path = session.get('temp_file_path')
    file_name = session.get('file_name')
    sheet_name = session.get('sheet_name')
    selected_rows = session.get('selected_rows', [])
    selected_cols = session.get('selected_cols', [])

    if not file_path or not file_name:
        return redirect('/')

    try:
        if file_name.endswith('.csv'):
            df = robust_read_csv(file_path)
        else:
            df = robust_read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        return render_template("upload.html", error=f"Error reloading file: {e}")

    df = df.fillna('').astype(str)
    mapping = {k.strip(): v for k, v in confirmed.items() if v.strip()}

    row_indices = [int(r.split('-')[1]) for r in selected_rows if r.startswith('row-')]
    col_indices = [int(c.split('-')[1]) for c in selected_cols if c.startswith('col-')]
    include_index = 'index' in selected_rows
    include_header = 'header' in selected_cols

    df_cleaned = df.copy()

    selected_row_labels = df_cleaned.index if not row_indices else [df.index[i] for i in row_indices]
    selected_col_labels = df_cleaned.columns if not col_indices else [df.columns[i] for i in col_indices]

    for row_idx in selected_row_labels:
        for col in selected_col_labels:
            val = df_cleaned.at[row_idx, col]
            df_cleaned.at[row_idx, col] = mapping.get(val.strip(), val)

    if include_index:
        df_cleaned.index = [
            mapping.get(str(idx).strip(), idx) for idx in df_cleaned.index
        ]
    if include_header:
        df_cleaned.columns = [
            mapping.get(str(col).strip(), col) for col in df_cleaned.columns
        ]

    # --- Generate in memory ---
    buffer = io.StringIO()
    df_cleaned.to_csv(buffer, index=True)
    buffer.seek(0)

    # Download filename
    base = os.path.splitext(secure_filename(file_name))[0]
    download_name = f"{base}_replaced.csv"

    return send_file(
        io.BytesIO(buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=download_name
    )


@app.route('/download_mapping')
def download_mapping():
    """Download a CSV of the original to matched name mapping."""
    confirmed = session.get('confirmed_mappings', {})
    file_name = session.get('file_name', 'mapping.csv')

    mapping_df = pd.DataFrame(
        [(orig, match) for orig, match in confirmed.items()],
        columns=['original_name', 'matched_name']
    )

    buffer = io.StringIO()
    mapping_df.to_csv(buffer, index=False)
    buffer.seek(0)

    base = os.path.splitext(secure_filename(file_name))[0]
    download_name = f"{base}_mapping.csv"

    return send_file(
        io.BytesIO(buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=download_name
    )



@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    """
    Custom handler for files >30MB
    """
    return render_template("upload.html", error="File too large. Please upload a file under 30MB."), 413


if __name__ == '__main__':
    app.run(debug=False)
