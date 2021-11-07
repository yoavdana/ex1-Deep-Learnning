# ex1_DL

Programing Task: Antigen Discovery for SARS-CoV-2 (“Corona”) Virus Vaccine

Time to end the COVID-19 pandemic by finding potential antigens. The latter are sub-sequences of the virus proteins that can be recognized by our immune system. Our adaptive immune system consists of 6 HLA (class I) alleles that allow it to selectively identify small fragments of proteins, known as peptides. The system is evolved to recognize only peptides of a foreign body and by that invoke an immune proliferation and response of T-cells that destroy the intruder. However, unfortunately not all foreign peptides are recognized. For those of you who are interested in learning more about this mechanism, I suggest this Wiki page.

In this exercise you will train a deep neural network to identify the peptides detected by a specific HLA allele known as HLA_A0201 which is a very common allele shared by 24% of the western population. The training data consists of ~3,000 positive and ~24,500 negative peptides. Each peptide consists of 9 amino acids (of 20 types). At a second stage, you will use your trained predictor to identify sequences of 9-amino-acids peptides from the Spike protein of the SARS-CoV-2 virus. 

Formally,

You will find the training data at the course’s moodle page. 
Set up a multi-layered perceptron network to accept this data and output the proper prediction (detect / not detect). Try different architectural changes (e.g., different number of levels, neurons at each level, etc.), and non-linearities (RelU, sigmoid) and pick the one achieving the highest accuracy on the test set. Document the tests you conducted in the submission, as well as the best performing architecture. 
Load the data from the files, and map it to the proper mathematical representation of your choice. Split it into 90%/10% train/test sets (picked at random at each run to avoid over-fitting). Explain your choice of representation in the report. Notice that the training data contains much more negative examples than positive. Use some machine learning strategy to deal with this form of unbalance and avoid convergence to trivial solution such as all negative predictions.
Train the network till convergence of the (train/test) loss plots. Make sure your learning rates are not too small and certainly not too large. Detail the chosen parameters in your document and add the train/test loss plots to the submission pdf.
Once you find your best configuration, plot the train and test metrics, and add it to your submission, in your report explain what methods you have tried.
Use your model to predict the detection of 9-amino-acids peptides from the Spike protein of the SARS-CoV-2, that is: all its consecutive 9-mer segments out of its 1273-amino-acid sequence. You can download this sequence from this page: https://viralzone.expasy.org/8996
Notify the CDC of the 5 most detectable peptides in this protein, as well as include them in your report.
Find the most detectable peptides according to your trained network by optimising the networks output score with respect to the input sequence name this function optimize_sequence  . For this purpose you should:
the input tensor should be declared as a trainable variables, by something like: w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)

And, make sure your optimizer operates on these variables and not your network’s weights.
Note that the solution to this problem can be a tensor of arbitrary nature (i.e., have arbitrary values and signs). So consider the way this solution is mapped into a meaningful peptide sequence. Report this resulting sequence.



