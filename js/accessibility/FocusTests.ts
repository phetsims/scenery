// Copyright 2017-2023, University of Colorado Boulder

/**
 * Focus tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import { Display, FocusManager, Node, Trail } from '../imports.js';

QUnit.module( 'Focus' );

type EqualityItem = {
  trail?: Trail;
  children: Node[];
};

type NestedEqualityItem = {
  trail?: Trail;
  children: NestedEqualityItem[];
};

// Arrays of items of the type { trail: {Trail}, children: {Array.<Item>} }
function nestedEquality( assert: Assert, a: EqualityItem[], b: NestedEqualityItem[] ): void {
  assert.equal( a.length, b.length );

  for ( let i = 0; i < a.length; i++ ) {
    const aItem = a[ i ];
    const bItem = b[ i ];

    if ( aItem.trail && bItem.trail ) {
      assert.ok( aItem.trail.equals( bItem.trail ) );
    }

    nestedEquality( assert, aItem.children, bItem.children );
  }
}

QUnit.test( 'Simple Test', assert => {

  const a1 = new Node( { tagName: 'div' } );
  const a2 = new Node( { tagName: 'div' } );

  const b1 = new Node( { tagName: 'div' } );
  const b2 = new Node( { tagName: 'div' } );

  const a = new Node( { children: [ a1, a2 ] } );
  const b = new Node( { children: [ b1, b2 ] } );

  const root = new Node( { children: [ a, b ] } );

  const nestedOrder = root.getNestedPDOMOrder();

  nestedEquality( assert, nestedOrder, [
    { trail: new Trail( [ root, a, a1 ] ), children: [] },
    { trail: new Trail( [ root, a, a2 ] ), children: [] },
    { trail: new Trail( [ root, b, b1 ] ), children: [] },
    { trail: new Trail( [ root, b, b2 ] ), children: [] }
  ] );
} );

QUnit.test( 'pdomOrder Simple Test', assert => {

  const a1 = new Node( { tagName: 'div' } );
  const a2 = new Node( { tagName: 'div' } );

  const b1 = new Node( { tagName: 'div' } );
  const b2 = new Node( { tagName: 'div' } );

  const a = new Node( { children: [ a1, a2 ] } );
  const b = new Node( { children: [ b1, b2 ] } );

  const root = new Node( { children: [ a, b ], pdomOrder: [ b, a ] } );

  const nestedOrder = root.getNestedPDOMOrder();

  nestedEquality( assert, nestedOrder, [
    { trail: new Trail( [ root, b, b1 ] ), children: [] },
    { trail: new Trail( [ root, b, b2 ] ), children: [] },
    { trail: new Trail( [ root, a, a1 ] ), children: [] },
    { trail: new Trail( [ root, a, a2 ] ), children: [] }
  ] );
} );

QUnit.test( 'pdomOrder Descendant Test', assert => {

  const a1 = new Node( { tagName: 'div' } );
  const a2 = new Node( { tagName: 'div' } );

  const b1 = new Node( { tagName: 'div' } );
  const b2 = new Node( { tagName: 'div' } );

  const a = new Node( { children: [ a1, a2 ] } );
  const b = new Node( { children: [ b1, b2 ] } );

  const root = new Node( { children: [ a, b ], pdomOrder: [ a1, b1, a2, b2 ] } );

  const nestedOrder = root.getNestedPDOMOrder();

  nestedEquality( assert, nestedOrder, [
    { trail: new Trail( [ root, a, a1 ] ), children: [] },
    { trail: new Trail( [ root, b, b1 ] ), children: [] },
    { trail: new Trail( [ root, a, a2 ] ), children: [] },
    { trail: new Trail( [ root, b, b2 ] ), children: [] }
  ] );
} );

QUnit.test( 'pdomOrder Descendant Pruning Test', assert => {

  const a1 = new Node( { tagName: 'div' } );
  const a2 = new Node( { tagName: 'div' } );

  const b1 = new Node( { tagName: 'div' } );
  const b2 = new Node( { tagName: 'div' } );

  const c1 = new Node( { tagName: 'div' } );
  const c2 = new Node( { tagName: 'div' } );

  const c = new Node( { children: [ c1, c2 ] } );

  const a = new Node( { children: [ a1, a2, c ] } );
  const b = new Node( { children: [ b1, b2 ] } );

  const root = new Node( { children: [ a, b ], pdomOrder: [ c1, a, a2, b2 ] } );

  const nestedOrder = root.getNestedPDOMOrder();

  nestedEquality( assert, nestedOrder, [
    { trail: new Trail( [ root, a, c, c1 ] ), children: [] },
    { trail: new Trail( [ root, a, a1 ] ), children: [] },
    { trail: new Trail( [ root, a, c, c2 ] ), children: [] },
    { trail: new Trail( [ root, a, a2 ] ), children: [] },
    { trail: new Trail( [ root, b, b2 ] ), children: [] },
    { trail: new Trail( [ root, b, b1 ] ), children: [] }
  ] );
} );

QUnit.test( 'pdomOrder Descendant Override', assert => {

  const a1 = new Node( { tagName: 'div' } );
  const a2 = new Node( { tagName: 'div' } );

  const b1 = new Node( { tagName: 'div' } );
  const b2 = new Node( { tagName: 'div' } );

  const a = new Node( { children: [ a1, a2 ] } );
  const b = new Node( { children: [ b1, b2 ], pdomOrder: [ b1, b2 ] } );

  const root = new Node( { children: [ a, b ], pdomOrder: [ b, b1, a ] } );

  const nestedOrder = root.getNestedPDOMOrder();

  nestedEquality( assert, nestedOrder, [
    { trail: new Trail( [ root, b, b2 ] ), children: [] },
    { trail: new Trail( [ root, b, b1 ] ), children: [] },
    { trail: new Trail( [ root, a, a1 ] ), children: [] },
    { trail: new Trail( [ root, a, a2 ] ), children: [] }
  ] );
} );

QUnit.test( 'pdomOrder Hierarchy', assert => {

  const a1 = new Node( { tagName: 'div' } );
  const a2 = new Node( { tagName: 'div' } );

  const b1 = new Node( { tagName: 'div' } );
  const b2 = new Node( { tagName: 'div' } );

  const a = new Node( { children: [ a1, a2 ], pdomOrder: [ a2 ] } );
  const b = new Node( { children: [ b1, b2 ], pdomOrder: [ b2, b1 ] } );

  const root = new Node( { children: [ a, b ], pdomOrder: [ b, a ] } );

  const nestedOrder = root.getNestedPDOMOrder();

  nestedEquality( assert, nestedOrder, [
    { trail: new Trail( [ root, b, b2 ] ), children: [] },
    { trail: new Trail( [ root, b, b1 ] ), children: [] },
    { trail: new Trail( [ root, a, a2 ] ), children: [] },
    { trail: new Trail( [ root, a, a1 ] ), children: [] }
  ] );
} );

QUnit.test( 'pdomOrder DAG test', assert => {

  const a1 = new Node( { tagName: 'div' } );
  const a2 = new Node( { tagName: 'div' } );

  const a = new Node( { children: [ a1, a2 ], pdomOrder: [ a2, a1 ] } );
  const b = new Node( { children: [ a1, a2 ], pdomOrder: [ a1, a2 ] } );

  const root = new Node( { children: [ a, b ] } );

  const nestedOrder = root.getNestedPDOMOrder();

  nestedEquality( assert, nestedOrder, [
    { trail: new Trail( [ root, a, a2 ] ), children: [] },
    { trail: new Trail( [ root, a, a1 ] ), children: [] },
    { trail: new Trail( [ root, b, a1 ] ), children: [] },
    { trail: new Trail( [ root, b, a2 ] ), children: [] }
  ] );
} );

QUnit.test( 'pdomOrder DAG test', assert => {

  const x = new Node();
  const a = new Node();
  const b = new Node();
  const c = new Node();
  const d = new Node( { tagName: 'div' } );
  const e = new Node();
  const f = new Node( { tagName: 'div' } );
  const g = new Node( { tagName: 'div' } );
  const h = new Node( { tagName: 'div' } );
  const i = new Node( { tagName: 'div' } );
  const j = new Node( { tagName: 'div' } );
  const k = new Node( { tagName: 'div' } );
  const l = new Node();

  x.children = [ a ];
  a.children = [ k, b, c ];
  b.children = [ d, e ];
  c.children = [ e ];
  e.children = [ j, f, g ];
  f.children = [ h, i ];

  x.pdomOrder = [ f, c, d, l ];
  a.pdomOrder = [ c, b ];
  e.pdomOrder = [ g, f, j ];

  const nestedOrder = x.getNestedPDOMOrder();

  nestedEquality( assert, nestedOrder, [
    // x order's F
    {
      trail: new Trail( [ x, a, b, e, f ] ), children: [
        { trail: new Trail( [ x, a, b, e, f, h ] ), children: [] },
        { trail: new Trail( [ x, a, b, e, f, i ] ), children: [] }
      ]
    },
    {
      trail: new Trail( [ x, a, c, e, f ] ), children: [
        { trail: new Trail( [ x, a, c, e, f, h ] ), children: [] },
        { trail: new Trail( [ x, a, c, e, f, i ] ), children: [] }
      ]
    },

    // X order's C
    { trail: new Trail( [ x, a, c, e, g ] ), children: [] },
    { trail: new Trail( [ x, a, c, e, j ] ), children: [] },

    // X order's D
    { trail: new Trail( [ x, a, b, d ] ), children: [] },

    // X everything else
    { trail: new Trail( [ x, a, b, e, g ] ), children: [] },
    { trail: new Trail( [ x, a, b, e, j ] ), children: [] },
    { trail: new Trail( [ x, a, k ] ), children: [] }
  ] );
} );

QUnit.test( 'setting pdomOrder', assert => {

  const rootNode = new Node();
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  const a = new Node( { tagName: 'div' } );
  const b = new Node( { tagName: 'div' } );
  const c = new Node( { tagName: 'div' } );
  const d = new Node( { tagName: 'div' } );
  rootNode.children = [ a, b, c, d ];

  // reverse accessible order
  rootNode.pdomOrder = [ d, c, b, a ];

  assert.ok( display._rootPDOMInstance, 'should exist' );

  const divRoot = display._rootPDOMInstance!.peer!.primarySibling!;
  const divA = a.pdomInstances[ 0 ].peer!.primarySibling;
  const divB = b.pdomInstances[ 0 ].peer!.primarySibling;
  const divC = c.pdomInstances[ 0 ].peer!.primarySibling;
  const divD = d.pdomInstances[ 0 ].peer!.primarySibling;

  assert.ok( divRoot.children[ 0 ] === divD, 'divD should be first child' );
  assert.ok( divRoot.children[ 1 ] === divC, 'divC should be second child' );
  assert.ok( divRoot.children[ 2 ] === divB, 'divB should be third child' );
  assert.ok( divRoot.children[ 3 ] === divA, 'divA should be fourth child' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'setting pdomOrder before setting accessible content', assert => {
  const rootNode = new Node();
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  const a = new Node();
  const b = new Node();
  const c = new Node();
  const d = new Node();
  rootNode.children = [ a, b, c, d ];

  // reverse accessible order
  rootNode.pdomOrder = [ d, c, b, a ];

  a.tagName = 'div';
  b.tagName = 'div';
  c.tagName = 'div';
  d.tagName = 'div';

  const divRoot = display._rootPDOMInstance!.peer!.primarySibling!;
  const divA = a.pdomInstances[ 0 ].peer!.primarySibling;
  const divB = b.pdomInstances[ 0 ].peer!.primarySibling;
  const divC = c.pdomInstances[ 0 ].peer!.primarySibling;
  const divD = d.pdomInstances[ 0 ].peer!.primarySibling;

  assert.ok( divRoot.children[ 0 ] === divD, 'divD should be first child' );
  assert.ok( divRoot.children[ 1 ] === divC, 'divC should be second child' );
  assert.ok( divRoot.children[ 2 ] === divB, 'divB should be third child' );
  assert.ok( divRoot.children[ 3 ] === divA, 'divA should be fourth child' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'setting accessible order on nodes with no accessible content', assert => {
  const rootNode = new Node();
  var display = new Display( rootNode ); // eslint-disable-line no-var
  document.body.appendChild( display.domElement );

  // root
  //    a
  //      b
  //     c   e
  //        d  f

  const a = new Node( { tagName: 'div' } );
  const b = new Node( { tagName: 'div' } );
  const c = new Node( { tagName: 'div' } );
  const d = new Node( { tagName: 'div' } );
  const e = new Node( { tagName: 'div' } );
  const f = new Node( { tagName: 'div' } );
  rootNode.addChild( a );
  a.addChild( b );
  b.addChild( c );
  b.addChild( e );
  c.addChild( d );
  c.addChild( f );
  b.pdomOrder = [ e, c ];

  const divB = b.pdomInstances[ 0 ].peer!.primarySibling!;
  const divC = c.pdomInstances[ 0 ].peer!.primarySibling!;
  const divE = e.pdomInstances[ 0 ].peer!.primarySibling!;

  assert.ok( divB.children[ 0 ] === divE, 'div E should be first child of div B' );
  assert.ok( divB.children[ 1 ] === divC, 'div C should be second child of div B' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );

} );

QUnit.test( 'setting accessible order on nodes with no accessible content', assert => {
  const rootNode = new Node();
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const a = new Node( { tagName: 'div' } );
  const b = new Node();
  const c = new Node( { tagName: 'div' } );
  const d = new Node( { tagName: 'div' } );
  const e = new Node( { tagName: 'div' } );
  const f = new Node( { tagName: 'div' } );
  rootNode.addChild( a );
  a.addChild( b );
  b.addChild( c );
  b.addChild( e );
  c.addChild( d );
  c.addChild( f );
  a.pdomOrder = [ e, c ];

  const divA = a.pdomInstances[ 0 ].peer!.primarySibling!;
  const divC = c.pdomInstances[ 0 ].peer!.primarySibling!;
  const divE = e.pdomInstances[ 0 ].peer!.primarySibling!;

  assert.ok( divA.children[ 0 ] === divE, 'div E should be first child of div B' );
  assert.ok( divA.children[ 1 ] === divC, 'div C should be second child of div B' );
  display.dispose();
  display.domElement.parentElement!.removeChild( display.domElement );
} );

QUnit.test( 'Testing FocusManager.windowHasFocusProperty', assert => {
  const rootNode = new Node();
  const display = new Display( rootNode );
  document.body.appendChild( display.domElement );

  const focusableNode = new Node( { tagName: 'button' } );
  rootNode.addChild( focusableNode );

  assert.ok( !FocusManager.windowHasFocusProperty.value, 'should not have focus at start' );

  // First, test detachFromWindow, once focus is in the window it is impossible to remove it from
  // the window with JavaScript.
  FocusManager.attachToWindow();
  FocusManager.detachFromWindow();

  assert.ok( !FocusManager.windowHasFocusProperty.value, 'should not have focus after detaching' );
  focusableNode.focus();
  assert.ok( !FocusManager.windowHasFocusProperty.value, 'Should not be watching window focus changes after detaching' );

  // now test changes to windowHasFocusProperty - window focus listeners will only work if tests are being run
  // in the foreground (dev cannot be using dev tools, running in puppeteer, minimized, etc...)
  if ( document.hasFocus() ) {
    FocusManager.attachToWindow();
    assert.ok( FocusManager.windowHasFocusProperty.value, 'Focus was moved into window from previous tests, this attach should reflect window already has focus.' );
    focusableNode.focus();
    assert.ok( FocusManager.windowHasFocusProperty.value, 'Window has focus, is now in the foreground' );
    focusableNode.blur();
    assert.ok( FocusManager.windowHasFocusProperty.value, 'window still has focus after a blur (focus on body)' );
  }

  FocusManager.detachFromWindow();
} );