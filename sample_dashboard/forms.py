from django import forms

class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)
    algorithm = forms.CheckboxSelectMultiple(choices=('polynomial regression','Polynomic kernel SVR','RBF kernel SVR'))