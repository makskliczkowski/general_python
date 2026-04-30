Contributing
============

Contributions to **GenUtils Python** are welcome and greatly appreciated! If you have an idea for an improvement or have found a bug, please consider contributing to the project.

**How to Contribute:**

1. **Fork the Repository:** Click the "Fork" button on the GitHub repository page to create your own copy of the project.
2. **Create a Feature Branch:** Clone your fork and create a new branch for your feature or bug fix:
   
   .. code-block:: bash

       git clone https://github.com/<your-username>/general_python.git
       cd general_python
       git checkout -b my-feature-branch

3. **Implement Your Changes:** Make your changes or additions. Follow the coding style of the project (for Python, adhere to PEP 8 guidelines). Ensure that you include docstrings for any new public functions or classes to maintain the documentation quality. See the docstring guidelines below.
4. **Write Tests (if applicable):** If the project contains a test suite, add tests for your new feature or bug fix. Ensuring that all tests pass will help your contribution get accepted more easily.
5. **Update Documentation:** Update or add documentation in this Sphinx docs (in the appropriate .rst files) if your changes affect the usage or API. This helps others understand your contribution.
6. **Commit and Push:** Commit your changes with clear and descriptive commit messages, then push your branch to your fork on GitHub.
7. **Open a Pull Request:** Go to the original repository on GitHub and open a pull request from your fork's new branch. Provide a clear description of your changes and any relevant context. The maintainers will review your contribution and provide feedback.

**Contribution Guidelines:**

- For larger changes or new features, it's recommended to open an issue first to discuss your ideas with the project maintainers. This discussion can provide guidance and ensure that your work aligns with the project's goals.
- Ensure your code passes any existing continuous integration checks (if configured) and does not decrease test coverage.
- By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.

Docstring Guidelines
--------------------

``general_python`` uses NumPy-style docstrings because they render cleanly in
Sphinx and are familiar to scientific Python users.

Public modules should start with a short summary, followed by a small overview
of the module's scope. Prefer user-facing language over implementation history
or file metadata.

Public functions and methods should document:

- Parameters, including expected shape and backend where relevant.
- Returns, including scalar versus array output and shape.
- Raises for common validation failures.
- Notes for numerical stability, randomness, backend differences, or expensive operations.
- Examples only when a short example clarifies non-obvious usage.

Classes should describe the invariant they maintain and the workflow they
support. Dataclasses and enums should have concise class docstrings even when
their fields are self-documenting.

Avoid placeholder docstrings such as "Get value" or "Initialize object". If the
behavior is obvious, explain the contract that matters to a library user:
accepted inputs, side effects, cached state, or compatibility aliases.
