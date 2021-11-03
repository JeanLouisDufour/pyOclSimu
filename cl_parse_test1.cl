enum
{
    hsv_shift  = 12,
	xxx,
	yyy = 36
};

enum foo
{
    foo_bar  = 12
};

typedef enum
{
    _bar  = 12
} toto, *titi;

__kernel void foo(__global int *bar) {
int i;
int isz = sizeof(i);
}

