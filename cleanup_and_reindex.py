"""
Utility script: cleans tmp* junk forms from SQLite and re-embeds all
real forms that have free-text content into Chroma.
"""
import sys
sys.path.insert(0, 'src')

import structured_store, vectorstore
from chunking import build_chunks
from schemas import FREE_TEXT_FIELD, FORM_SCHEMAS
from extraction import ExtractedForm

# 1) Clean tmp* junk from SQLite
conn = structured_store.connect()
cur = conn.execute("SELECT form_id, form_type FROM forms_index WHERE form_id LIKE 'tmp%'")
junk = cur.fetchall()
print('Removing junk forms:', [r[0] for r in junk])
for row in junk:
    fid, ftype = row['form_id'], row['form_type']
    conn.execute(f"DELETE FROM form_{ftype} WHERE form_id = ?", (fid,))
    conn.execute('DELETE FROM forms_index WHERE form_id = ?', (fid,))
conn.commit()
print('SQLite cleaned.')
print('Remaining IDs:', structured_store.all_form_ids(conn))

# 2) Re-embed all real forms that have free-text content
client = vectorstore.get_client()
col = vectorstore.get_collection(client)
print(f'\nChroma chunk count before: {vectorstore.count_chunks(col)}')

for fid in structured_store.all_form_ids(conn):
    form = structured_store.get_form(conn, fid)
    ftype = form.get('form_type')
    if ftype not in FREE_TEXT_FIELD:
        print(f'  {fid}: unknown form_type {ftype!r}, skipping')
        continue
    ft_field = FREE_TEXT_FIELD[ftype]
    ft_text = form.get(ft_field, '') or ''
    if not ft_text.strip():
        print(f'  {fid}: no free-text in {ft_field!r}, skipping')
        continue

    schema_cls = FORM_SCHEMAS[ftype]
    data_fields = {k: form.get(k) for k in schema_cls.model_fields if k in form}
    extracted = ExtractedForm(
        form_id=fid,
        form_type=ftype,
        data=schema_cls(**data_fields),
        detection_method='db_reingest',
        extraction_attempts=0,
    )
    vectorstore.delete_form(col, fid)
    written = vectorstore.add_chunks(col, build_chunks(extracted))
    print(f'  {fid}: wrote {written} chunk(s)')

print(f'\nChroma chunk count after: {vectorstore.count_chunks(col)}')
print('Done.')
