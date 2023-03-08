// Copyright 2023, University of Colorado Boulder

/**
 * QUnit tests for MatrixBetweenProperty
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import { MatrixBetweenProperty, Node } from '../imports.js';

QUnit.module( 'MatrixBetweenProperty' );

QUnit.test( 'MatrixBetweenProperty connectivity', assert => {

  const a = new Node();
  const b = new Node();
  const c = new Node();
  const x = new Node();
  const y = new Node();

  const matrixBetweenProperty = new MatrixBetweenProperty( x, y );

  const checkMatrix = ( matrix: Matrix3 | null, message: string ) => {
    const propMatrix = matrixBetweenProperty.value;
    assert.ok( propMatrix === matrix || ( matrix && propMatrix && matrix.equals( propMatrix ) ), message );
  };

  checkMatrix( null, 'no connection at all' );

  // a -> x
  a.addChild( x );
  checkMatrix( null, 'no connection at all' );

  //   x
  //  /
  // a
  //  \
  //   y
  a.addChild( y );
  checkMatrix( Matrix3.IDENTITY, 'connected, identity' );

  // b    x
  //  \  /
  //   a
  //  /  \
  // c    y
  b.addChild( a );
  c.addChild( a );
  checkMatrix( Matrix3.IDENTITY, 'ignores DAG below, identity' );

  // b -> x
  //  \  /
  //   a
  //  /  \
  // c -> y
  b.addChild( x );
  c.addChild( y );
  checkMatrix( null, 'DAGs (cax/bax/bx, so null' );

  // b -> x
  //  \
  //   a
  //  /  \
  // c -> y
  a.removeChild( x );
  checkMatrix( Matrix3.IDENTITY, 'ignores DAG from C, since it is not reachable to x, identity' );

  // b
  //  \
  //   a
  //  /  \
  // c -> y
  //  \
  //   x
  b.removeChild( x );
  c.addChild( x );
  checkMatrix( null, 'DAG cay/cy, so null' );

  // b
  //  \
  //   a
  //  /  \
  // c -> y -> x
  c.removeChild( x );
  y.addChild( x );
  checkMatrix( Matrix3.IDENTITY, 'direct child OK' );

  //   a
  //  /  \
  // c -> y -> b -> x
  b.removeChild( a );
  y.removeChild( x );
  b.addChild( x );
  y.addChild( b );
  checkMatrix( Matrix3.IDENTITY, 'ancestor OK' );

  //   a------
  //     \    \
  // c -> y -> b -> x
  c.removeChild( a );
  a.addChild( b );
  checkMatrix( null, 'DAG aybx/abx, null' );

  //         a
  //          \
  // c -> y -> b -> x
  a.removeChild( y );
  checkMatrix( Matrix3.IDENTITY, 'back to normal' );

  matrixBetweenProperty.dispose();
} );

QUnit.test( 'MatrixBetweenProperty transforms (local)', assert => {

  const a = new Node();
  const b = new Node();
  const x = new Node();
  const y = new Node();

  const matrixBetweenProperty = new MatrixBetweenProperty( x, y );

  const checkMatrix = ( matrix: Matrix3 | null, message: string ) => {
    const propMatrix = matrixBetweenProperty.value;
    assert.ok( propMatrix === matrix || ( matrix && propMatrix && matrix.equals( propMatrix ) ), `message expected\n${matrix}\n\ngot\n${propMatrix}` );
  };

  checkMatrix( null, 'no connection at all' );

  //   x
  //  /
  // a
  //  \
  //   y
  a.addChild( x );
  a.addChild( y );
  checkMatrix( Matrix3.IDENTITY, 'connected, identity' );

  //   x (x:50)
  //  /
  // a
  //  \
  //   y
  x.x = 50;
  checkMatrix( Matrix3.rowMajor(
    1, 0, 50,
    0, 1, 0,
    0, 0, 1
  ), 'connected, 50 translation' );

  //   x (x:50)
  //  /
  // a
  //  \
  //   y (scale:2)
  y.scale( 2 );
  checkMatrix( Matrix3.rowMajor(
    0.5, 0, 25,
    0, 0.5, 0,
    0, 0, 1
  ), 'connected, 50 translation + 2 scale' );

  //   x (x:50)
  //  /
  // a (x:-50)
  //  \
  //   y (scale:2)
  a.x = -50;
  checkMatrix( Matrix3.rowMajor(
    0.5, 0, 25,
    0, 0.5, 0,
    0, 0, 1
  ), 'parent translation should not affect things' );

  //     x (x:50)
  //    /
  //   a (x:-50)
  //  /
  // b
  //  \
  //   y (scale:2)
  a.removeChild( y );
  b.addChild( a );
  b.addChild( y );
  checkMatrix( Matrix3.rowMajor(
    0.5, 0, 0,
    0, 0.5, 0,
    0, 0, 1
  ), 'now 50 and -50 cancel each other out' );

  //     x (x:50)
  //    /
  //   a (x:-50, y:10)
  //  /
  // b
  //  \
  //   y (scale:2)
  a.y = 10;
  checkMatrix( Matrix3.rowMajor(
    0.5, 0, 0,
    0, 0.5, 5,
    0, 0, 1
  ), 'adjusting transform on an ancestor' );

  //       x (x:50)
  //      /
  //     a (x:-50, y:10)
  //    /
  //   b
  //  /
  // y (scale:2)
  b.removeChild( y );
  y.addChild( b );
  checkMatrix( Matrix3.rowMajor(
    1, 0, 0,
    0, 1, 10,
    0, 0, 1
  ), 'swapping to no common root, instead an ancestor (ignores y transform)' );

  //       y (scale:2)
  //      /
  //     x (x:50)
  //    /
  //   a (x:-50, y:10)
  //  /
  // b
  y.removeChild( b );
  x.addChild( y );
  checkMatrix( Matrix3.rowMajor(
    0.5, 0, 0,
    0, 0.5, 0,
    0, 0, 1
  ), 'swapping order' );
} );

QUnit.test( 'MatrixBetweenProperty transforms (parent)', assert => {

  const a = new Node();
  const b = new Node();
  const x = new Node();
  const y = new Node();

  const matrixBetweenProperty = new MatrixBetweenProperty( x, y, {
    fromCoordinateFrame: 'parent',
    toCoordinateFrame: 'parent'
  } );

  const checkMatrix = ( matrix: Matrix3 | null, message: string ) => {
    const propMatrix = matrixBetweenProperty.value;
    assert.ok( propMatrix === matrix || ( matrix && propMatrix && matrix.equals( propMatrix ) ), `${message} expected\n${matrix}\n\ngot\n${propMatrix}` );
  };

  checkMatrix( null, 'no connection at all' );

  //   x
  //  /
  // a
  //  \
  //   y
  a.addChild( x );
  a.addChild( y );
  checkMatrix( Matrix3.IDENTITY, 'connected, identity' );

  //   x (x:50)
  //  /
  // a
  //  \
  //   y
  x.x = 50;
  checkMatrix( Matrix3.IDENTITY, 'x/y transforms do not matter #1' );

  //   x (x:50)
  //  /
  // a
  //  \
  //   y (scale:2)
  y.scale( 2 );
  checkMatrix( Matrix3.IDENTITY, 'x/y transforms do not matter #2' );

  //   x (x:50)
  //  /
  // a (x:-50)
  //  \
  //   y (scale:2)
  a.x = -50;
  checkMatrix( Matrix3.IDENTITY, 'x/y transforms do not matter #3' );

  //     x (x:50)
  //    /
  //   a (x:-50)
  //  /
  // b
  //  \
  //   y (scale:2)
  a.removeChild( y );
  b.addChild( a );
  b.addChild( y );
  checkMatrix( Matrix3.rowMajor(
    1, 0, -50,
    0, 1, 0,
    0, 0, 1
  ), 'now the -50 applies' );

  //     x (x:50)
  //    /
  //   a (x:-50, y:10)
  //  /
  // b
  //  \
  //   y (scale:2)
  a.y = 10;
  checkMatrix( Matrix3.rowMajor(
    1, 0, -50,
    0, 1, 10,
    0, 0, 1
  ), 'adjusting transform on an ancestor' );

  //       x (x:50)
  //      /
  //     a (x:-50, y:10)
  //    /
  //   b
  //  /
  // y (scale:2)
  b.removeChild( y );
  y.addChild( b );
  checkMatrix( Matrix3.rowMajor(
    2, 0, -100,
    0, 2, 20,
    0, 0, 1
  ), 'swapping to no common root, instead an ancestor' );

  //       y (scale:2)
  //      /
  //     x (x:50)
  //    /
  //   a (x:-50, y:10)
  //  /
  // b
  y.removeChild( b );
  x.addChild( y );
  checkMatrix( Matrix3.rowMajor(
    1, 0, -50,
    0, 1, 0,
    0, 0, 1
  ), 'swapping order' );
} );
