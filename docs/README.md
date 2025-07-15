# Future location of website doc templates
## To quickly setup locally...
**Prerequisites to running script**: installing python and pip.

```bash
make docs # or equivalently `bash local-init.sh`.
```
## To Update Website...
Make updates in the `mkdocs.yml` file (docs: https://www.mkdocs.org/user-guide/configuration/). If theme updates needed go to `base.yml` (docs: https://squidfunk.github.io/mkdocs-material/setup/).

## Further Customizations.
Can be done within the [`.theme/home.html`](./theme/home.html) file to modify the home page look.

## To Update pages.
Make the edits in the subdirectory README and the updates will be reflected in the github pages site after a commit push to main (or merge).

## To Add New Plugins (i.e. pages).
1. Create and update README.md within the specific plugin subdirectory in the repo (`mcp/src/...`).
2. Create associated plugin named `.md` file in `mcp/docs/docs/servers/...` (e.g. `mcp/src/core` -> `mcp/docs/docs/servers/core.md`).
3. Add this to the `.md` file created. Filling in the name of the plugin subdirectory in the `src/` folder.
    ```
    {% include "../../../src/{plugin-name-here}/README.md" %}
    ```
4. Update the [`mkdocs.yml`](./mkdocs.yml) `nav` section by adding the page to the main site navigation (under "Servers" sections).
