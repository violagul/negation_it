from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, GPT2Tokenizer

tokenizer = AutoTokenizer.from_pretrained("LorenzoDeMattei/GePpeTto")
model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto")




### prompt mask-like task, 2 examples, one neg target and one positive target
### in my test this gave verbs but copied from previous sentences
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
### in my test this didn't continue the prompts with verbs
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



### prompt mask-like task, 10 examples, one neg target and one positive target
### in my test this didn't continue the prompts with verbs
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
prompts = [
    "Nella frase 'Sara è una studentessa che ha l'abitudine di esagerare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'dormire'. Nella frase 'Luisa è una studentessa che ha l'abitudine di lavorare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'lavorare'. Nella frase 'Viola è una studentessa che ha l'abitudine di disegnare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'indicare'. Nella frase 'Alessandra è una studentessa che ha l'abitudine di meditare. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'meditare'. Nella frase 'Sara è una studentessa che ha l'abitudine di suonare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'ascoltare'. Nella frase 'Sara è una studentessa che ha l'abitudine di camminare. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'camminare'. Nella frase 'Sonia è una studentessa che ha l'abitudine di mangiare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'saltare'. Nella frase 'Elisa è una studentessa che ha l'abitudine di scrivere. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'scrivere'. Nella frase 'Anna è una studentessa che ha l'abitudine di cantare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'giocare'. Nella frase 'Giulio è un meccanico che ha l'abitudine di studiare. Lui X molto spesso.', si potrebbe sostituire X  con il verbo 'studiare'. Nella frase 'Elena è una ballerina che ha l'abitudine di leggere. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo",
    "Nella frase 'Sara è una studentessa che ha l'abitudine di esagerare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'dormire'. Nella frase 'Luisa è una studentessa che ha l'abitudine di lavorare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'lavorare'. Nella frase 'Viola è una studentessa che ha l'abitudine di disegnare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'indicare'. Nella frase 'Alessandra è una studentessa che ha l'abitudine di meditare. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'meditare'. Nella frase 'Sara è una studentessa che ha l'abitudine di suonare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'ascoltare'. Nella frase 'Sara è una studentessa che ha l'abitudine di camminare. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'camminare'. Nella frase 'Sonia è una studentessa che ha l'abitudine di mangiare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'saltare'. Nella frase 'Elisa è una studentessa che ha l'abitudine di scrivere. Lei X molto spesso.', si potrebbe sostituire X  con il verbo 'scrivere'. Nella frase 'Anna è una studentessa che ha l'abitudine di cantare. Lei non X molto spesso.', si potrebbe sostituire X  con il verbo 'giocare'. Nella frase 'Giulio è un meccanico che ha l'abitudine di studiare. Lui X molto spesso.', si potrebbe sostituire X  con il verbo 'studiare'. Nella frase 'Elena è una ballerina che ha l'abitudine di leggere. Lei X molto spesso.', si potrebbe sostituire X  con il verbo"]
    

samples_outputs = text_generator(
    prompts,
    do_sample=True,
    max_length=395,
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