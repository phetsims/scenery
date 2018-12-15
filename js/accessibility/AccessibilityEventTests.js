// Copyright 2017, University of Colorado Boulder

/**
 * Accessibility tests
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  const Display = require( 'SCENERY/display/Display' );
  const Node = require( 'SCENERY/nodes/Node' );
  const Rectangle = require( 'SCENERY/nodes/Rectangle' );

  // constants
  const TEST_LABEL = 'Test Label';
  const TEST_LABEL_2 = 'Test Label 2';

  QUnit.module( 'AccessibilityEvents' );

  const dispatchEvent = ( domElement, event ) => {
    domElement.dispatchEvent( new window.Event( event, {
      'bubbles': true // that is vital to all that scenery events hold near and dear to their hearts.
    } ) );
  };

  QUnit.test( 'focusin/focusout (focus/blur)', assert => {

    // TODO: remove this test, it must be committed to test on CT, see https://github.com/phetsims/phet-io-wrappers/issues/217
    setTimeout( () => {
      throw new Error( 'test' );
    }, 1000 );

    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    const a = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    let aGotFocus = false;
    let aLostFocus = false;
    let bGotFocus = false;

    rootNode.addChild( a );

    a.addInputListener( {
      focus() {
        aGotFocus = true;
      },
      blur() {
        aLostFocus = true;
      }
    } );

    a.focus();
    assert.ok( aGotFocus, 'a should have been focused' );
    assert.ok( !aLostFocus, 'a should not blur' );

    const b = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    // TODO: what if b was child of a, make sure these events don't bubble!
    rootNode.addChild( b );

    b.addInputListener( {
      focus() {
        bGotFocus = true;
      }
    } );

    b.focus();

    assert.ok( bGotFocus, 'b should have been focused' );
    assert.ok( aLostFocus, 'a should have lost focused' );
  } );

  QUnit.test( 'click', assert => {


    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    const a = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    let gotFocus = false;
    let gotClick = false;
    let aClickCounter = 0;

    rootNode.addChild( a );

    a.addInputListener( {
      focus() {
        gotFocus = true;
      },
      click() {
        gotClick = true;
        aClickCounter++;
      },
      blur() {
        gotFocus = false;
      }
    } );


    a.accessibleInstances[ 0 ].peer.primarySibling.focus();
    assert.ok( gotFocus && !gotClick, 'focus first' );
    a.accessibleInstances[ 0 ].peer.primarySibling.click(); // this works because it's a button
    assert.ok( gotClick && gotFocus && aClickCounter === 1, 'a should have been clicked' );

    let bClickCounter = 0;

    const b = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    b.addInputListener( {
      click() {
        bClickCounter++;
      }
    } );

    a.addChild( b );

    b.accessibleInstances[ 0 ].peer.primarySibling.focus();
    b.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( bClickCounter === 1 && aClickCounter === 2, 'a should have been clicked with b' );
    a.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( bClickCounter === 1 && aClickCounter === 3, 'b still should not have been clicked.' );


    // create a node
    const a1 = new Node( {
      tagName: 'button'
    } );
    a.addChild( a1 );
    assert.ok( a1.inputListeners.length === 0, 'no input accessible listeners on instantiation' );
    assert.ok( a1.labelContent === null, 'no label on instantiation' );

    // add a listener
    const listener = { click() { a1.labelContent = TEST_LABEL; } };
    a1.addInputListener( listener );
    assert.ok( a1.inputListeners.length === 1, 'accessible listener added' );

    // verify added with hasInputListener
    assert.ok( a1.hasInputListener( listener ) === true, 'found with hasInputListener' );

    // fire the event
    a1.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( a1.labelContent === TEST_LABEL, 'click fired, label set' );

    const c = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );
    const d = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );
    const e = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    let cClickCount = 0;
    let dClickCount = 0;
    let eClickCount = 0;

    rootNode.addChild( c );
    c.addChild( d );
    d.addChild( e );

    c.addInputListener( {
      click() {
        cClickCount++;
      }
    } );
    d.addInputListener( {
      click() {
        dClickCount++;
      }
    } );
    e.addInputListener( {
      click() {
        eClickCount++;
      }
    } );

    e.accessibleInstances[ 0 ].peer.primarySibling.click();

    assert.ok( cClickCount === dClickCount && cClickCount === eClickCount && cClickCount === 1,
      'click should have bubbled to all parents' );

    d.accessibleInstances[ 0 ].peer.primarySibling.click();


    assert.ok( cClickCount === 2 && dClickCount === 2 && eClickCount === 1,
      'd should not trigger click on e' );
    c.accessibleInstances[ 0 ].peer.primarySibling.click();


    assert.ok( cClickCount === 3 && dClickCount === 2 && eClickCount === 1,
      'c should not trigger click on d or e' );

    // reset click count
    cClickCount = 0;
    dClickCount = 0;
    eClickCount = 0;

    c.accessibleOrder = [ d, e ];

    e.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( cClickCount === 1 && dClickCount === 0 && eClickCount === 1,
      'accessibleOrder means click should bypass d' );

    c.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( cClickCount === 2 && dClickCount === 0 && eClickCount === 1,
      'click c should not effect e or d.' );

    d.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( cClickCount === 3 && dClickCount === 1 && eClickCount === 1,
      'click d should not effect e.' );

    // reset click count
    cClickCount = 0;
    dClickCount = 0;
    eClickCount = 0;

    const f = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    let fClickCount = 0;
    f.addInputListener( {
      click() {
        fClickCount++;
      }
    } );
    e.addChild( f );

    // so its a chain in the scene graph c->d->e->f

    d.accessibleOrder = [ f ];

    /* accessible instance tree:
         c
        / \
        d  e
        |
        f
    */

    f.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( cClickCount === 1 && dClickCount === 1 && eClickCount === 0 && fClickCount === 1,
      'click d should not effect e.' );
  } );


  QUnit.test( 'click extra', assert => {

    // create a node
    const a1 = new Node( {
      tagName: 'button'
    } );
    const root = new Node( { tagName: 'div' } );
    const display = new Display( root ); // eslint-disable-line

    // need to initializeEvents to add input listeners
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    root.addChild( a1 );
    assert.ok( a1.inputListeners.length === 0, 'no input accessible listeners on instantiation' );
    assert.ok( a1.labelContent === null, 'no label on instantiation' );

    // add a listener
    const listener = { click: function() { a1.labelContent = TEST_LABEL; } };
    a1.addInputListener( listener );
    assert.ok( a1.inputListeners.length === 1, 'accessible listener added' );

    // verify added with hasInputListener
    assert.ok( a1.hasInputListener( listener ) === true, 'found with hasInputListener' );

    // fire the event
    a1.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( a1.labelContent === TEST_LABEL, 'click fired, label set' );

    // remove the listener
    a1.removeInputListener( listener );
    assert.ok( a1.inputListeners.length === 0, 'accessible listener removed' );

    // verify removed with hasInputListener
    assert.ok( a1.hasInputListener( listener ) === false, 'not found with hasInputListener' );

    // make sure event listener was also removed from DOM element
    // click should not change the label
    a1.labelContent = TEST_LABEL_2;
    assert.ok( a1.labelContent === TEST_LABEL_2, 'before click' );

    // setting the label redrew the pdom, so get a reference to the new dom element.
    a1.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( a1.labelContent === TEST_LABEL_2, 'click should not change label' );

    // verify disposal removes accessible input listeners
    a1.addInputListener( listener );
    a1.dispose();

    // TODO: Since converting to use Node.inputListeners, we can't assume this anymore
    // assert.ok( a1.hasInputListener( listener ) === false, 'disposal removed accessible input listeners' );
  } );

  QUnit.test( 'input', assert => {

    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    const a = new Rectangle( 0, 0, 20, 20, { tagName: 'input', inputType: 'text' } );

    let gotFocus = false;
    let gotInput = false;

    rootNode.addChild( a );

    a.addInputListener( {
      focus() {
        gotFocus = true;
      },
      input() {
        gotInput = true;
      },
      blur() {
        gotFocus = false;
      }
    } );

    a.accessibleInstances[ 0 ].peer.primarySibling.focus();
    assert.ok( gotFocus && !gotInput, 'focus first' );

    dispatchEvent( a.accessibleInstances[ 0 ].peer.primarySibling, 'input' );

    assert.ok( gotInput && gotFocus, 'a should have been an input' );

  } );


  QUnit.test( 'change', assert => {


    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    const a = new Rectangle( 0, 0, 20, 20, { tagName: 'input', inputType: 'range' } );

    let gotFocus = false;
    let gotChange = false;

    rootNode.addChild( a );

    a.addInputListener( {
      focus() {
        gotFocus = true;
      },
      change() {
        gotChange = true;
      },
      blur() {
        gotFocus = false;
      }
    } );

    a.accessibleInstances[ 0 ].peer.primarySibling.focus();
    assert.ok( gotFocus && !gotChange, 'focus first' );

    dispatchEvent( a.accessibleInstances[ 0 ].peer.primarySibling, 'change' );

    assert.ok( gotChange && gotFocus, 'a should have been an input' );
  } );

  QUnit.test( 'keydown/keyup', assert => {


    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    const a = new Rectangle( 0, 0, 20, 20, { tagName: 'input', inputType: 'text' } );

    let gotFocus = false;
    let gotKeydown = false;
    let gotKeyup = false;

    rootNode.addChild( a );

    a.addInputListener( {
      focus() {
        gotFocus = true;
      },
      keydown() {
        gotKeydown = true;
      },
      keyup() {
        gotKeyup = true;
      },
      blur() {
        gotFocus = false;
      }
    } );

    a.accessibleInstances[ 0 ].peer.primarySibling.focus();
    assert.ok( gotFocus && !gotKeydown, 'focus first' );

    dispatchEvent( a.accessibleInstances[ 0 ].peer.primarySibling, 'keydown' );

    assert.ok( gotKeydown && gotFocus, 'a should have had keydown' );

    dispatchEvent( a.accessibleInstances[ 0 ].peer.primarySibling, 'keyup' );
    assert.ok( gotKeydown && gotKeyup && gotFocus, 'a should have had keyup' );
  } );

} );