# include <iostream>
using namespace std;

    struct node{
        int data;
        node*next;
    }
 int main(){
    node* head=new node();
    head->data=10;
    head->next=NULL;
    node*first=new node();
    first->data=20;
    first->next=NULL;
    node*second=new node();
    second->data=30;
    second->next=NULL;
    head->next=first;
    first->next=second;
   second->next=NULL;
   node*temp=head;
   while(temp!=NULL){
    cout<<temp->data<<" ";
    temp=temp->next;
   }


 }