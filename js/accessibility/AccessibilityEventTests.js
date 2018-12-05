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
  var Display = require( 'SCENERY/display/Display' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );

  QUnit.module( 'AccessibilityEvents' );

  QUnit.test( 'focusin/focusout', assert => {


    var rootNode = new Node( { tagName: 'div' } );
    var display = new Display( rootNode ); // eslint-disable-line
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    var a = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    var aGotFocus = false;
    var aLostFocus = false;
    var bGotFocus = false;

    rootNode.addChild( a );

    a.addInputListener( {
      focus() {
        console.log( 'afocus' );

        aGotFocus = true;
      },
      blur() {
        console.log( 'ablur' );

        aLostFocus = true;
      }
    } );

    a.focus();
    assert.ok( aGotFocus, 'a should have been focused' );
    assert.ok( !aLostFocus, 'a should not blur' );

    var b = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    // TODO: what if b was child of a, make sure these events don't bubble!
    rootNode.addChild( b );

    b.addInputListener( {
      focus() {
        console.log( 'bfocus' );

        bGotFocus = true;
      }
    } );

    b.focus();

    assert.ok( bGotFocus, 'b should have been focused' );
    assert.ok( aLostFocus, 'a should have lost focused' );
  } );

  QUnit.test( 'click', assert => {


    let rootNode = new Node( { tagName: 'div' } );
    let display = new Display( rootNode ); // eslint-disable-line
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    let a = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

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

    let b = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

    a.addInputListener( {
      click() {
        bClickCounter++;
      }
    } );

    a.addChild( b );

    b.accessibleInstances[ 0 ].peer.primarySibling.focus();
    b.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( bClickCounter === 1 && aClickCounter === 2, 'a should have been clicked' );


    // create a node
    var a1 = new Node( {
      tagName: 'button'
    } );
    a.addChild( a1 );
    assert.ok( a1.inputListeners.length === 0, 'no input accessible listeners on instantiation' );
    assert.ok( a1.labelContent === null, 'no label on instantiation' );

    // add a listener
    var listener = { click() { a1.labelContent = TEST_LABEL; } };
    a1.addInputListener( listener );
    assert.ok( a1.inputListeners.length === 1, 'accessible listener added' );

    // verify added with hasInputListener
    assert.ok( a1.hasInputListener( listener ) === true, 'found with hasInputListener' );

    // fire the event
    a1.accessibleInstances[ 0 ].peer.primarySibling.click();
    assert.ok( a1.labelContent === TEST_LABEL, 'click fired, label set' );

  } );


  QUnit.test( 'click extra', assert => {

    // create a node
    var a1 = new Node( {
      tagName: 'button'
    } );
    let root = new Node( { tagName: 'div' } );
    var display = new Display( root ); // eslint-disable-line

    // need to initializeEvents to add input listeners
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    root.addChild( a1 );
    assert.ok( a1.inputListeners.length === 0, 'no input accessible listeners on instantiation' );
    assert.ok( a1.labelContent === null, 'no label on instantiation' );

    // add a listener
    var listener = { click: function() { a1.labelContent = TEST_LABEL; } };
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


    let rootNode = new Node( { tagName: 'div' } );
    let display = new Display( rootNode ); // eslint-disable-line
    display.initializeEvents();
    document.body.appendChild( display.domElement );

    let a = new Rectangle( 0, 0, 20, 20, { tagName: 'input', inputType: 'text' } );

    let gotFocus = false;
    let gotInput = false;

    rootNode.addChild( a );

    a.addInputListener( {
      focus() {
        console.log( 'in focus' );

        gotFocus = true;
      },
      input() {
        console.log( 'in input;' );
        gotInput = true;
      },
      blur() {
        console.log( 'in blur' );
        gotFocus = false;
      }
    } );

    console.log( 'before focus call' );
    a.accessibleInstances[ 0 ].peer.primarySibling.focus();
    console.log( 'after focus call' );
    assert.ok( gotFocus && !gotInput, 'focus first' );
    a.accessibleInstances[ 0 ].peer.primarySibling.dispatchEvent( new window.Event( 'input', {
      'bubbles': true
    } ) );
    console.log( 'after input call' );

    assert.ok( gotInput && gotFocus, 'a should have been an input' );

  } );
} );