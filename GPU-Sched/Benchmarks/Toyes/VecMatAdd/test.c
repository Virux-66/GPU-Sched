extern int callee(int x);
int f(int x) {
	  int result = (x / 42);
	  callee(result);
	    return result;
}
