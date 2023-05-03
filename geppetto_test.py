from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, GPT2Tokenizer

tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto")




### prompt mask-like task, 2 examples, one neg target and one positive target
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
prompts = [
    "Nella frase 'Anna è una studentessa che ha l'abitudine di cantare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'giocare'. Nella frase 'Giulio è un meccanico che ha l'abitudine di studiare. Lui X molto spesso.', si potrebbe sostituire X  con il verbo 'studiare'. Nella frase 'Elena è una ballerina che ha l'abitudine di leggere. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo",
    "Nella frase 'Anna è una studentessa che ha l'abitudine di cantare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'giocare'. Nella frase 'Giulio è un meccanico che ha l'abitudine di studiare. Lui X molto spesso.', si potrebbe sostituire X  con il verbo 'studiare'. Nella frase 'Elena è una ballerina che ha l'abitudine di leggere. Lei X molto spesso.', si potrebbe sostituire X  con il verbo"]


samples_outputs = text_generator(
    prompts,
    do_sample=True,
    max_length=110,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3
)

for i, sample_outputs in enumerate(samples_outputs):
    print(100 * '-')
    print("Prompt:", prompts[i])
    for sample_output in sample_outputs:
        print("Sample:", sample_output['generated_text'])
        print()




### prompt mask-like task, 6 examples, one neg target and one positive target
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
prompts = [
    "Nella frase 'Sara è una studentessa che ha l'abitudine di suonare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'ascoltare'. Nella frase 'Sara è una studentessa che ha l'abitudine di camminare. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'camminare'. Nella frase 'Sonia è una studentessa che ha l'abitudine di mangiare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'saltare'. Nella frase 'Elisa è una studentessa che ha l'abitudine di scrivere. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'scrivere'. Nella frase 'Anna è una studentessa che ha l'abitudine di cantare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'giocare'. Nella frase 'Giulio è un meccanico che ha l'abitudine di studiare. Lui X molto spesso.', si potrebbe sostituire X  con il verbo 'studiare'. Nella frase 'Elena è una ballerina che ha l'abitudine di leggere. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo",
    "Nella frase 'Sara è una studentessa che ha l'abitudine di suonare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'ascoltare'. Nella frase 'Sara è una studentessa che ha l'abitudine di camminare. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'camminare'. Nella frase 'Sonia è una studentessa che ha l'abitudine di mangiare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'saltare'. Nella frase 'Elisa è una studentessa che ha l'abitudine di scrivere. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'scrivere'. Nella frase 'Anna è una studentessa che ha l'abitudine di cantare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'giocare'. Nella frase 'Giulio è un meccanico che ha l'abitudine di studiare. Lui X molto spesso.', si potrebbe sostituire X  con il verbo 'studiare'. Nella frase 'Elena è una ballerina che ha l'abitudine di leggere. Lei X molto spesso.', si potrebbe sostituire X  con il verbo"]
    

samples_outputs = text_generator(
    prompts,
    do_sample=True,
    max_length=250,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3
)

for i, sample_outputs in enumerate(samples_outputs):
    print(100 * '-')
    print("Prompt:", prompts[i])
    for sample_output in sample_outputs:
        print("Sample:", sample_output['generated_text'])
        print()