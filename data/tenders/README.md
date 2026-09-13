# Tenders

One folder per tender. The pipeline takes a tender folder as an argument; nothing
about a specific tender is hardcoded anywhere in the code.

    data/tenders/<tender_name>/
        documents/    the tender exactly as the client issued it — PDF, XLSX, DOCX
        gold/         answer key. Only exists for a tender that has been labelled
                      for evaluation. A real tender has no gold folder.

To run a real tender: create the folder, drop the client's files into
`documents/`, point the pipeline at the tender folder. That is the whole setup.

No pipeline module may read anything under a `gold/` folder. A test enforces it.
