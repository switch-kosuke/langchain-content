def test_run_llm_non_english_query():
    # Arrange
    query = "¿Qué es un enlace de cadena de LangChain?"  # Spanish query
    expected_answer_language = "Spanish"

    # Act
    result = run_llm(query)

    # Assert
    assert result["answer"].split(" ")[0] == expected_answer_language, "Answer should be in Spanish"