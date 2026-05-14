## Instalación de uv

Compruebe que tiene instalado Python en su sistema. Después siga las instrucciones de instalación de [`uv`](https://github.com/astral-sh/uv).


## Instalación de `Quarto` con `uv`

Ejecute:

```bash 
uv tool install quarto
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
