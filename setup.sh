mkdir -p ~/.streamlit/

echo -e "\
[general]\n\
email = \"fabiomacedomendes@gmail.com\"\n\
"

echo -e "\
[general]\n\
email = \"fabiomacedomendes@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo -e "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
"

echo -e "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml