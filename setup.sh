mkdir -p ~/.streamlit/echo “\
[general]\n\
email = \”fabiomacedomendes@gmail.com\”\n\
“ > ~/.streamlit/credentials.tomlecho “\

[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
“ > ~/.streamlit/config.toml