mkdir -p ~/.streamlit/
printf "\
[general]\n\
email = \"fabiomacedomendes@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
cat ~/.streamlit/credentials.toml

printf "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
cat ~/.streamlit/config.toml