mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"info@basinclimbing.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[theme]\n\
base = \"light\"\n\
primaryColor = \"#8B4229\"\n\
backgroundColor = \"#FFFFFF\"\n\
secondaryBackgroundColor = \"#F5F3EE\"\n\
textColor = \"#213B3F\"\n\
font = \"sans serif\"\n\
\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
