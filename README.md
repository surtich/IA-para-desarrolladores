## Instalación de uv

Compruebe que tiene instalado Python en su sistema. Después siga las instrucciones de instalación de [`uv`](https://github.com/astral-sh/uv).

## Instalación de la `tool` de `quarto-cli`

```bash 
uv tool install --from quarto-cli quarto-cli
```

## Instalación de dependencias del proyecto

Sitúese en el directorio raíz del proyecto y ejecute:

```bash 
uv sync
```

## Para ejecutar el modo preview de `Quarto`

```bash 
uv run quarto preview
```
