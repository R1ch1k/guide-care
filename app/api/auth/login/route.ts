import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { password } = await request.json();

    // Get the password from environment variable
    const correctPassword = process.env.TEAM_PASSWORD || 'changeme123';

    if (password === correctPassword) {
      // Create response with auth cookie
      const response = NextResponse.json({ success: true });

      // Set HTTP-only cookie that expires in 7 days
      response.cookies.set('auth-token', 'authenticated', {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        maxAge: 60 * 60 * 24 * 7, // 7 days
        path: '/',
      });

      return response;
    } else {
      return NextResponse.json(
        { error: 'Invalid password' },
        { status: 401 }
      );
    }
  } catch (error) {
    return NextResponse.json(
      { error: 'Invalid request' },
      { status: 400 }
    );
  }
}
