author = 'John Bogaardt, et. al.'
bibtex_bibfiles = ['references.bib']
comments_config = {'hypothesis': False, 'utterances': False}
copyright = '2021'
exclude_patterns = ['**.ipynb_checkpoints', '.DS_Store', 'Thumbs.db', '_build', 'templates']
execution_allow_errors = False
execution_excludepatterns = []
execution_in_temp = False
execution_timeout = 30
extensions = ['sphinx_togglebutton', 'sphinx_copybutton', 'myst_nb', 'jupyter_book', 'sphinx_thebe', 'sphinx_comments', 'sphinx_external_toc', 'sphinx.ext.intersphinx', 'sphinx_panels', 'sphinx_book_theme', 'sphinx_gallery.gen_gallery', 'sphinx_gallery.load_style', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'numpydoc', 'sphinx.ext.mathjax', 'sphinxcontrib.bibtex', 'sphinx_jupyterbook_latex']
external_toc_exclude_missing = False
external_toc_path = 'C:/Users/jboga/Documents/github/chainladder-python/docs/_toc.yml'
html_add_permalinks = '¶'
html_baseurl = ''
html_favicon = ''
html_logo = 'logo.png'
html_sourcelink_suffix = ''
html_theme = 'sphinx_book_theme'
html_theme_options = {'search_bar_text': 'Search this book...', 'launch_buttons': {'notebook_interface': 'classic', 'binderhub_url': 'https://mybinder.org', 'jupyterhub_url': '', 'thebe': False, 'colab_url': ''}, 'path_to_docs': 'docs', 'repository_url': 'https://github.com/casact/chainladder-python', 'repository_branch': 'master', 'google_analytics_id': '', 'extra_navbar': 'Powered by <a href="https://jupyterbook.org">Jupyter Book</a>', 'extra_footer': '', 'home_page_in_toc': True, 'use_repository_button': True, 'use_edit_page_button': False, 'use_issues_button': True}
html_title = 'Reserving in Python'
jupyter_cache = ''
jupyter_execute_notebooks = 'force'
language = None
latex_engine = 'pdflatex'
myst_enable_extensions = ['colon_fence', 'dollarmath', 'linkify', 'substitution']
myst_url_schemes = ['mailto', 'http', 'https']
nb_output_stderr = 'show'
numfig = True
panels_add_bootstrap_css = False
pygments_style = 'sphinx'
sphinx_gallery_conf = {'doc_module': 'chainladder', 'backreferences_dir': 'modules\\\\generated', 'first_notebook_cell': "import matplotlib.pyplot as plt\nplt.style.use('ggplot')\n%config InlineBackend.figure_format = 'svg'", 'reference_url': {'chainladder': None}}
suppress_warnings = ['myst.domains']
use_jupyterbook_latex = True
use_multitoc_numbering = True

import chainladder as cl
version = cl.__version__
release = cl.__version__
