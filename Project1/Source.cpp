#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
void eulerProblem7();
void eulerProblem8();
void eulerProblem9();
void eulerProblem30();
void eulerProblem26();
void eulerProblem17();
void eulerProblem69();
void eulerProblem_69();
void testPrimesVsPrimes2();
int calcTotient(int n, std::vector<int> primes);
std::vector<int> getPrimes(int n);
std::vector<int> getPrimes2(int n);
int main() {
	eulerProblem9();
	
	std::cin.get();
	return 0;
}

void eulerProblem26(){
	int sequence[100000];
	int sequenceLength = 0;
	bool notInSequence = true;
	int topSequenceLength = 0;
	for (int i = 1; i < 1000; i++) {
		std::cout << "onNumber: " << i << std::endl;
		sequenceLength = 0;
		notInSequence = true;
		int divisor = 1;
		int remainder = 0;
		int topNumber = divisor / i;
		while (notInSequence) {
			topNumber = divisor / i;
			remainder = divisor - i * topNumber;
			if (remainder == 0) break;
			else {
				sequence[sequenceLength] = remainder;
				divisor = remainder * 10;
				for (int j = 0; j < sequenceLength; j++) {
					if (remainder == sequence[j]) {
						notInSequence = false;
						
						if (sequenceLength - j > topSequenceLength) {
							std::cout << "highestNumber: " << i << "at: " << sequenceLength - j << std::endl;
							topSequenceLength = sequenceLength - j;
						}
					}
				}
				sequenceLength++;
			}
		}
	}
	std::cout << topSequenceLength;
}

void eulerProblem7() {
	int counter = 1;
	int currentNum = 3;
	bool notPrime = true;
	const int primeToFind = 100001;
	int primeList[primeToFind];
	primeList[0] = 2;
	while (counter<primeToFind) {
		notPrime = false;
		for (int i = 0; i < counter && primeList[i] <= std::sqrt(currentNum); i++) {
			if (currentNum % primeList[i] == 0) {
				notPrime = true;
				i = currentNum;
			}
		}
		if (!notPrime) {
			primeList[counter] = currentNum;
			counter++;
		}
		currentNum++;
	}
	std::cout << currentNum - 1 << std::endl;
}

void eulerProblem30() {
	int answer = 0;
	int power = 5;
	for (int i = 2; i < std::pow(9, power)*(power+1); i++) {
		int digitCounter = 0;
		int temp = i;
		for (int j = 0; j < power+1; j++) {
			digitCounter += std::pow(temp % 10,power);
			temp /= 10;
		}
		if (digitCounter == i) {
			answer += digitCounter;
		}
	}
	std::cout << "answer: " << answer << std::endl;
}

void eulerProblem17() {
	std::string onesWords[10] = {"","one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
	std::string teensWords[10] = {"ten","eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
	std::string tensWords[8] = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
	std::string hundredWord = "hundred";
	std::string andWord = "and";
	std::string oneThousand = "oneThousand";

	int numberLength = 0;
	int totalLength = 0;
	for(int temp = 1; temp < 1000; temp++){
		int ones = temp%10;
		int tens = (temp/10)%10;
		int hundreds = (temp/100)%10;
		
		
		
		numberLength=int(onesWords[hundreds].length() + (hundreds?(hundredWord.length()+((ones||tens)?andWord.length():0)):0));
		std::cout<<temp<<": "<<"Hundreds place "<<hundreds<<", "<<onesWords[hundreds]<<", length: "<<numberLength<<std::endl; //", "<<((ones||tens)?andWord.length():0)
		
		//std::cout<<onesWords[hundreds] << hundreds?(hundredWord+ones?andWord:""):""<<" ";
		switch(tens){
			case 0:
				numberLength+=onesWords[ones].length();
				std::cout<<onesWords[ones];
			break;
			case 1:
				numberLength+=teensWords[ones].length();
				std::cout<<teensWords[ones];
			break;
			default:
				numberLength+=tensWords[tens-2].length()+onesWords[ones].length();
				std::cout<<tensWords[tens-2] << onesWords[ones] <<std::endl;
			break;
		}

		std::cout<<numberLength<<std::endl;
		totalLength+=numberLength;
		//std::cout <<"ones place lengths "<<i<<": "<< onesPlace[0].length()<<std::endl;
		//std::cout <<"teens place lengths "<<i<<": "<< teens[0].length()<<std::endl;
	}

	std::cout<<totalLength + oneThousand.length()<<std::endl;
}

int gcd(int a, int b)  
{  
    if (a == 0)  
        return b;  
    return gcd(b % a, a);  
}  
  
// A simple method to evaluate Euler Totient Function  
int phi(unsigned int n)  
{  
    unsigned int result = 1;  
    for (int i = 2; i < n; i++)  
        if (gcd(i, n) == 1)  
            result++;  
    return result;  
}
void eulerProblem69(){
	double answer = 0;
	double temp=0;
	int solution = 0;
	auto t1 = std::chrono::high_resolution_clock::now();
	auto t3 = std::chrono::high_resolution_clock::now();
	for(int i = 2; i <1000000; i++){
		temp = double(i)/calcTotient(i,getPrimes2(i));
		if(temp>answer){
			answer=temp;
			solution = i;
		}
		if(i%10000==0){
			std::cout<<"answer: "<<answer<<", solution: "<<solution<<std::endl;
			auto t2 = std::chrono::high_resolution_clock::now();
			std::cout<<"took: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()<<"ms"<<std::endl;
			t1=t2;
			std::cout<<double(i)/10000<<"% done"<<std::endl;
		}
	}
	auto t4 = std::chrono::high_resolution_clock::now();
	std::cout<<"answer: "<<answer<<", solution: "<<solution<<std::endl;
	std::cout<<"took: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count()<<"ms"<<std::endl;
}

int calcTotient(int n, std::vector<int> primes){
	int compositeNums=0; 
	int tot = 0;
	int baseNum = 1;
	/*std::cout<<"n: "<<n<<", primes list: "<<std::endl;
	for(int i = 0; i < primes.size();i++){
		std::cout<<primes[i]<<", ";
	}
	std::cout<<std::endl;*/
	/*for(int i = 0; i < primes.size(); i++){
		tot+=n/primes[i];
		baseNum*=primes[i];
		for(int j = i+1; j < primes.size();j++){
			compositeNums++;
			tot-=(n/(primes[i]*primes[j]));
		}
	}*/
	//For each group size of primes find all combinations from size 1 to primes.size()-1
	int underCounting = -1;
	for(int g = 1; g <= primes.size(); g++){
		underCounting*=-1;
		int numToDevideBy = 1;
		std::string bitmask(g, 1); // K leading 1's
		bitmask.resize(primes.size(), 0); // N-K trailing 0's
	
		// print integers and permute bitmask
		do {
			numToDevideBy = 1;
			for (int i = 0; i < primes.size(); ++i) // [0..N-1] integers
			{
				if (bitmask[i]){
					//std::cout << " " << i<<","<<primes[i]<<" ";
					numToDevideBy*=primes[i];
				}
			}
			//std::cout <<std::endl<<" "<<underCounting*(n/numToDevideBy)<<" "<<numToDevideBy<< std::endl;
			tot+=underCounting*(n/numToDevideBy);
		} while (std::prev_permutation(bitmask.begin(), bitmask.end()));
	}
	//std::cout<<"total: "<<tot;
		//this pattern 
		// ooo--, oo-o-, oo--o, o-oo-, o-o-o, o--oo, -ooo-, -oo-o
		// for i=0; i <= primes.size()-groupSize; i++
	return n-tot;//n-(tot+(compositeNums*n/baseNum-(primes.size()-1)*n/baseNum));
}

std::vector<int> getPrimes(int n){
	std::vector<int> primesList;
	bool primesNotFound = true;
	int curNum = 2; 
	while(curNum<=n){
		bool repeat = false;
		while(n%curNum==0 && n>1){
			if(!repeat){
				primesList.push_back(curNum);
				repeat = true;
			}
			n/=curNum;
		}
		curNum++;
	}
	//primesList.push_back(n);
	//if(curNum==n) primesList.push_back(curNum);
	return primesList; 
}
std::vector<int> getPrimes2(int n){
	std::vector<int> primesList;
	bool primesNotFound = true;
	int curNum = 2; 
	while(curNum<sqrt(n)){
		bool repeat = false;
		while(n%curNum==0 && n>1){
			if(!repeat){
				primesList.push_back(curNum);
				repeat = true;
			}
			n/=curNum;
		}
		curNum++;
	}
	if(n>1){
		if(curNum==sqrt(n))
			n=curNum;
		primesList.push_back(n);

	}
		
	//if(curNum==n) primesList.push_back(curNum);
	return primesList; 
}

void eulerProblem_69(){
	/*for(int i = 2; i < 10000; i++){
		if(calcTotient(i,getPrimes(i)) != phi(i)){
			std::cout<<"New tot vs old tot "<<i<<std::endl;
			std::cout<<calcTotient(i,getPrimes(i))<<std::endl;
			std::cout<<phi(i)<<std::endl;
		}
	}*/
	double answer = 0;
	double temp=0;
	int solution = 0;
	unsigned long int phiTime = 0;
	unsigned long int totTime = 0;
	for(int i = 2; i <1000000; i++){
		auto t1 = std::chrono::high_resolution_clock::now();
		temp = double(i)/phi(i);
		auto t2 = std::chrono::high_resolution_clock::now();
		phiTime+=std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();

		t1 = std::chrono::high_resolution_clock::now();
		temp = double(i)/calcTotient(i, getPrimes(i));
		t2 = std::chrono::high_resolution_clock::now();
		totTime+=std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
		if(i%1000==0){
			std::cout<<"phiTime: "<<phiTime<<"ns"<<std::endl;
			std::cout<<"totTime: "<<totTime<<"ns"<<std::endl;
			std::cout<<double(i)/10000<<"% done"<<std::endl;
		}
	}
	std::cout<<"answer: "<<answer<<std::endl;
	std::cout<<"done"<<std::endl;
	/*for(int i = 0; i < 91; i+=2){
		std::cout<<i;
		
			std::cout<<", ";
		
	}
	std::cout<<std::endl;
	for(int i = 0; i < 91; i+=3){
		std::cout<<i;
		
			std::cout<<", ";
		
	}
	std::cout<<std::endl;
	for(int i = 0; i < 91; i+=5){
		std::cout<<i;
		
			std::cout<<", ";
		
	}*/
	//primes.push_back(2);
	//primes.push_back(3);
	//primes.push_back(5);
	/*std::cout<<"n: "<<n<<", primes list: "<<std::endl;
	for(int i = 0; i < primes.size();i++){
		std::cout<<primes[i]<<", ";
	}*/
	//std::cout<<"primes "<<calcTotient(n,primes)<<std::endl;
}
  
void testPrimesVsPrimes2(){
	std::vector<int> primes;
	std::vector<int> newPrimes;
	int numTests = 1000000;
	for(int i = 0; i < numTests; i++){
		primes = getPrimes(i);
		newPrimes = getPrimes2(i);
		if(i%(numTests/100)==0){
			std::cout<<i/(numTests/100)<<"% done"<<std::endl;
		}
		if(primes.size()!=newPrimes.size()){
			std::cout<<"number: "<<i<<std::endl;
			for(int j = 0; j < primes.size(); j++){
				std::cout<<" "<<primes[j];
			}
			std::cout<<std::endl;
			for(int j = 0; j < newPrimes.size(); j++){
				std::cout<<" "<<newPrimes[j];
			}
			std::cout<<std::endl;
		}
		else{
			bool samePrimes = true;
			for(int j = 0; j < primes.size(); j++){
				if(primes[j]!=newPrimes[j]) samePrimes=false;
			}
			if(!samePrimes){
				std::cout<<"number: "<<i<<std::endl;
				for(int j = 0; j < primes.size(); j++){
					std::cout<<" "<<primes[j];
				}
				std::cout<<std::endl;
				for(int j = 0; j < newPrimes.size(); j++){
					std::cout<<" "<<newPrimes[j];
				}
				std::cout<<std::endl;
			}
		}
	}
	std::cout<<"done"<<std::endl;
}

void eulerProblem8(){
	std::string numberString="7316717653133062491922511967442657474235534919493496983520312774506326239578318016984801869478851843858615607891129494954595017379583319528532088055111254069874715852386305071569329096329522744304355766896648950445244523161731856403098711121722383113622298934233803081353362766142828064444866452387493035890729629049156044077239071381051585930796086670172427121883998797908792274921901699720888093776657273330010533678812202354218097512545405947522435258490771167055601360483958644670632441572215539753697817977846174064955149290862569321978468622482839722413756570560574902614079729686524145351004748216637048440319989000889524345065854122758866688116427171479924442928230863465674813919123162824586178664583591245665294765456828489128831426076900422421902267105562632111110937054421750694165896040807198403850962455444362981230987879927244284909188845801561660979191338754992005240636899125607176060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530420752963450";
	unsigned long biggestNum = 0;
	unsigned long tempNum = 1;
	for(int j = 0; j < 13; j++){
		tempNum*=int(numberString[j])-48;
		//std::cout<<"string val: "<<numberString[i+j]<<"num val: "<<int(numberString[i+j])-48<<std::endl;
	}
	biggestNum = tempNum;
	std::cout<<tempNum<<std::endl;
	int bigOCounter =13;
	for(int i=13; i < numberString.length(); i++){
		if(int(numberString[i])-48==0){
			i+=12;
			tempNum=1;
		}
		else if(int(numberString[i-13])-48==0){
			tempNum=1;
			for(int j = 0; j < 13; j++){
				if(int(numberString[i-12+j])-48 == 0){
					i+=j+1;
					j=0;
					tempNum=1;
					
				}
				if(i>999) break;
				tempNum*=int(numberString[i-12+j])-48;
				bigOCounter++;
			}
			std::cout<<"?"<<tempNum<<std::endl;
			if(tempNum>biggestNum) biggestNum=tempNum;
		}
		else{
			tempNum/=int(numberString[i-13])-48;
			tempNum*=int(numberString[i])-48;
			bigOCounter++;
			std::cout<<tempNum<<std::endl;
			if(tempNum>biggestNum) biggestNum=tempNum;
		}
		
	}
	std::cout<<"biggest num: "<<biggestNum<<", "<<"bigO: "<<bigOCounter<<std::endl;
}


void eulerProblem9(){
	int totRun = 0;
	for(int a = 1; a < 293; a++){
		//a with the smallest possible b where a<b must fit a^2+b^2<=c^2 to have a chance
		//of finding a b and c that work 
		//a^2 + b^2 <= c^2
		//a^2 + (a+1)^2 <= (1000-(2a+1))^2
		//2a^2 + 2a + 1 <= (999-2a)^2
		//2a^2 + 2a + 1 <= 999^2 - 3996a + 4a^2
		//-2a^2 + 3998a <= 999^2 - 1
		//a<292
		//if(pow(((1000-(a+1))/2),2) + a*a < pow((1000-(a+1))/2,2))
		//a^{2}+x^{2}=(1000-a-x)^{2} because a is known we can find b from this
		//derived equation below \/
		int w=1000-a;
		int bStart = (a*a-w*w)/(-2*w);
		if((a*a-w*w)%(-2*w)==0){
			std::cout<<"done early, a: "<<a<<", b: "<<bStart<<", c: "<<1000-a-bStart<<std::endl;
			std::cout<<"answer: "<<a*bStart*(1000-a-bStart)<<", found in "<<totRun<<" steps"<<std::endl;
		}
		totRun++;
	}
	std::cout<<"done";
}