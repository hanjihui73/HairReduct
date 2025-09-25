#ifndef __PHYS_VECTOR_DYNAMIC_H__
#define __PHYS_VECTOR_DYNAMIC_H__

#include <math.h>
#include <stdio.h>

template <typename T>
class Vec3
{
public:
	T		_x;
	T		_y;
	T		_z;
public:
	Vec3(void)
	{
		_x = (T)0;
		_y = (T)0;
		_z = (T)0;
	}
	Vec3(Vec3 const &v)
	{
		_x = v._x;
		_y = v._y;
		_z = v._z;
	}
	Vec3(T const &x, T const &y, T const &z)
	{
		_x = x;
		_y = y;
		_z = z;
	}
	~Vec3() {}
	void						x(T d) { _x = d; }
	void						y(T d) { _y = d; }
	void						z(T d) { _z = d; }
	void						set(Vec3<T> v)
	{
		_x = v._x;
		_y = v._y;
		_z = v._z;
	}
	void						set(T x)
	{
		_x = x;
		_y = x;
		_z = x;
	}
	void						set(T x, T y, T z)
	{
		_x = x;
		_y = y;
		_z = z;
	}
	void						clear(void)
	{
		_x = 0.0;
		_y = 0.0;
		_z = 0.0;
	}
	void						print(void)
	{
		printf("%f, %f, %f\n", _x, _y, _z);
	}
	void						normalize(void)
	{
		double norm = getNorm();
		if (norm != 0) {
			_x = _x / norm;
			_y = _y / norm;
			_z = _z / norm;
		}
	}
	void						inverse(void)
	{
		_x *= -1.0;
		_y *= -1.0;
		_z *= -1.0;
	}
	T						getNorm(void)
	{
		return sqrt(_x*_x + _y * _y + _z * _z);
	}
	T						length(void)
	{
		return(T)(sqrt(lengthSquared()));
	}
	T						lengthSquared(void)
	{
		return (T)(_x*_x + _y * _y + _z * _z);
	}
	Vec3<T>						abs(void)
	{
		return Vec3(fabs(_x), fabs(_y), fabs(_z));
	}

	T						dot(Vec3<T>& v)
	{
		return (_x*v.x() + _y * v.y() + _z * v.z());
	}
	T						x(void) { return _x; }
	T						y(void) { return _y; }
	T						z(void) { return _z; }
	T						&get(int n)
	{
		return  *((&_x) + n);
	}
	Vec3<T>	cross(Vec3<T>& v)
	{
		Vec3<T> vector;
		vector.x((_y*v.z()) - (_z*v.y()));
		vector.y((_z*v.x()) - (_x*v.z()));
		vector.z((_x*v.y()) - (_y*v.x()));
		return vector;
	}
public:
	bool						operator==(Vec3 const &v) const
	{
		return _x == v._x && _y == v._y && _z == v._z;
	}
	bool						operator!=(Vec3 const & v) const
	{
		return _x != v._x && _y != v._y && _z != v._z;
	}
	T						&operator()(int index)
	{
		return  *((&_x) + index);
	}
	T						&operator[](int index)
	{
		return  *((&_x) + index);
	}
	T const					&operator()(int index) const
	{
		return  *((&_x) + index);
	}
	T const					&operator[](int index) const
	{
		return  *((&_x) + index);
	}
	Vec3<T>	ortho(void)
	{
		if (_x != 0) {
			return Vec3(-_y, _x, 0);
		}
		else {
			return Vec3(1, 0, 0);
		}
	}
	Vec3<T>	&operator=(Vec3 const &v)
	{
		_x = v._x;
		_y = v._y;
		_z = v._z;
		return *this;
	}
	Vec3<T>	&operator+=(Vec3 const &v)
	{
		_x += v._x;
		_y += v._y;
		_z += v._z;
		return *this;
	}
	Vec3<T>	&operator+=(T v)
	{
		_x += v;
		_y += v;
		_z += v;
		return *this;
	}
	Vec3<T>	&operator-=(T v)
	{
		_x -= v;
		_y -= v;
		_z -= v;
		return *this;
	}
	Vec3<T>	&operator-=(Vec3 const &v)
	{
		_x -= v._x;
		_y -= v._y;
		_z -= v._z;
		return *this;
	}
	Vec3<T>	&operator*=(T const &d)
	{
		_x *= d;
		_y *= d;
		_z *= d;
		return *this;
	}
	Vec3<T>	&operator*=(Vec3<T> &v)
	{
		_x *= v.x();
		_y *= v.y();
		_z *= v.z();
		return *this;
	}
	Vec3<T>	&operator/=(T const &d)
	{
		_x /= d;
		_y /= d;
		_z /= d;
		return *this;
	}
	Vec3<T>	operator/(const T &d)
	{
		return Vec3(_x / d, _y / d, _z / d);
	}
	Vec3<T>	operator*(const T &d)
	{
		return Vec3(_x*d, _y*d, _z*d);
	}
	Vec3<T>	operator-(const T &d)
	{
		return Vec3(_x - d, _y - d, _z - d);
	}
	Vec3<T>	operator+(const T &d)
	{
		return Vec3(_x + d, _y + d, _z + d);
	}
	Vec3<T>	operator-() const
	{
		return Vec3(-_x, -_y, -_z);
	}
	Vec3<T>	operator+(Vec3 const &v) const
	{
		return Vec3(_x + v._x, _y + v._y, _z + v._z);
	}
	Vec3<T>	operator-(Vec3 const &v) const
	{
		return Vec3(_x - v._x, _y - v._y, _z - v._z);
	}
	Vec3<T>	operator*(Vec3<T>	&v)
	{
		return Vec3(_x*v._x, _y*v._y, _z*v._z);
	}
	T						angle(Vec3<T> &v)
	{
		T _dot, _cross;
		Vec3<T> tmp;
		_dot = dot(v);  // Cos * norms
		tmp.set(cross(v));
		_cross = tmp.length(); //  Sin * norms
		return atan2(_cross, _dot);
	}

	Vec3<T> lerp(Vec3<T> q, double t)
	{
		Vec3<T> tmp;
		tmp = (*this) * t + (q * (1.0 - t));
		return tmp;
	}
};

typedef Vec3<double> vec3;

#endif