Training and Testing Dataset Ratio
Total number of records = 10000
Training = 70%
Testing = 30%

Total number of features = 48
Classifiers prediction rate 

Naive-Bayes accuracy :  0.837
LinearSVC accuracy :  0.9183333333333333
KNeighbors accuracy score :  0.8713333333333333

Filtered Top 20 Features via Chi-Squared
========================================
                          URL_features        Score
 47  PctExtNullSelfRedirectHyperlinksRT  3028.933077
 34          FrequentDomainNameMismatch  1971.202716
 4                              NumDash  1138.258721
 38                   SubmitInfoToEmail  1027.151102
 33       PctNullSelfRedirectHyperlinks   944.585365
 29                       InsecureForms   745.967495
 0                              NumDots   682.739637
 26                    PctExtHyperlinks   550.028317
 24                   NumSensitiveWords   505.983345
 39                       IframeOrFrame   399.649120
 2                            PathLevel   363.937885
 45              AbnormalExtFormActionR   221.524336
 43                         UrlLengthRT   199.513030
 20                      HostnameLength   194.873075
 5                    NumDashInHostname   168.570028
 10                  NumQueryComponents   154.485877
 25                   EmbeddedBrandName   152.045080
 32                  AbnormalFormAction   135.219340
 16                           IpAddress   120.645341
 18                       DomainInPaths   106.725700
 
Classifier Accuracy for top 10 filtered features
================================================
Naive-Bayes accuracy :  0.8177
LinearSVC accuracy :  0.8877
KNeighbors accuracy score :  0.9563

Classifier Accuracy for top 20 filtered features
================================================
Naive-Bayes accuracy :  0.8173
LinearSVC accuracy :  0.9207
KNeighbors accuracy score :  0.9187
 
 
Filtered Top 20 Features via Information Gain (IG) 
=================================================
PctExtHyperlinks
PctExtResourceUrls
PctNullSelfRedirectHyperlinks
PctExtNullSelfRedirectHyperlinksRT
NumNumericChars
FrequentDomainNameMismatch
ExtMetaScriptLinkRT
NumDash
SubmitInfoToEmail
NumDots


