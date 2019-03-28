// Copyright 2017, University of Colorado Boulder

/**
 * Tests related to Accessibility input and events.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  const KeyboardUtil = require( 'SCENERY/accessibility/KeyboardUtil' );
  const Display = require( 'SCENERY/display/Display' );
  const Node = require( 'SCENERY/nodes/Node' );
  const Rectangle = require( 'SCENERY/nodes/Rectangle' );

  // constants
  const TEST_LABEL = 'Test Label';
  const TEST_LABEL_2 = 'Test Label 2';

  QUnit.module( 'AccessibilityInput' );

  /**
   * Set up a test for accessible input by attaching a root node to a display and initializing events.
   * @param {Display} display
   */
  const beforeTest = ( display ) => {
    display.initializeEvents();
    document.body.appendChild( display.domElement );
  };

  /**
   * Clean up a test by detaching events and removing the element from the DOM so that it doesn't interfere
   * with QUnit UI.
   * @param {Display} display
   */
  const afterTest = ( display ) => {
    display.detachEvents();
    document.body.removeChild( display.domElement );
  };

  const dispatchEvent = ( domElement, event ) => {
    domElement.dispatchEvent( new window.Event( event, {
      'bubbles': true // that is vital to all that scenery events hold near and dear to their hearts.
    } ) );
  };

  // create a fake DOM event and delegate to an HTMLElement
  // TODO: Can this replace the dispatchEvent function above?
  const triggerDOMEvent = ( event, element, keyCode, options ) => {

    options = _.extend( {

      // secondary target for the event, behavior depends on event type
      relatedTarget: null
    }, options );

    const eventObj = document.createEventObject ?
                   document.createEventObject() : document.createEvent( 'Events' );

    if ( eventObj.initEvent ) {
      eventObj.initEvent( event, true, true );
    }

    eventObj.keyCode = keyCode;
    eventObj.which = keyCode;
    eventObj.relatedTarget = options.relatedTarget;

    element.dispatchEvent ? element.dispatchEvent( eventObj ) : element.fireEvent( 'on' + event, eventObj );
  };

  QUnit.test( 'focusin/focusout (focus/blur)', assert => {

    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    beforeTest( display );

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

    afterTest( display );
  } );

  QUnit.test( 'tab focusin/focusout', assert => {
    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    beforeTest( display );

    const buttonA = new Rectangle( 0, 0, 5, 5, { tagName: 'button' } );
    const buttonB = new Rectangle( 0, 0, 5, 5, { tagName: 'button' } );
    const buttonC = new Rectangle( 0, 0, 5, 5, { tagName: 'button' } );
    rootNode.children = [ buttonA, buttonB, buttonC ];

    const aPrimarySibling = buttonA.accessibleInstances[ 0 ].peer.primarySibling;
    const bPrimarySibling = buttonB.accessibleInstances[ 0 ].peer.primarySibling;

    // test that a blur listener on a node overides the "tab" like navigation moving focus to the next element
    buttonA.focus();
    assert.ok( buttonA.focused, 'butonA has focus initially' );

    const overrideFocusListener = {
      blur: function( event ) {
        buttonC.focus();
      }
    };
    buttonA.addInputListener( overrideFocusListener );

    // mimic a "tab" interaction, attempting to move focus to the next element
    triggerDOMEvent( 'focusout', aPrimarySibling, KeyboardUtil.KEY_TAB, {
      relatedTarget: bPrimarySibling
    } );

    // the blur listener on buttonA should override the movement of focus on "tab" like interaction
    assert.ok( buttonC.focused, 'butonC now has focus' );

    // test that a blur listener can prevent focus from moving to another element after "tab" like navigation
    buttonA.removeInputListener( overrideFocusListener );
    buttonA.focus();
    const makeUnfocusableListener = {
      blur: function( event ) {
        buttonB.focusable = false;
      }
    };
    buttonA.addInputListener( makeUnfocusableListener );
    triggerDOMEvent( 'focusout', aPrimarySibling, KeyboardUtil.KEY_TAB, {
      relatedTarget: bPrimarySibling
    } );

    // the blur listener on buttonA should have made the default element unfocusable
    assert.ok( !buttonB.focused, 'buttonB cannot receive focus due to blur listener on buttonA' );

    afterTest( display );
  } );

  QUnit.test( 'click', assert => {

    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    beforeTest( display );

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

    afterTest( display );
  } );

  QUnit.test( 'click extra', assert => {

    // create a node
    const a1 = new Node( {
      tagName: 'button'
    } );
    const root = new Node( { tagName: 'div' } );
    const display = new Display( root ); // eslint-disable-line
    beforeTest( display );

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
    
    afterTest( display );
  } );

  QUnit.test( 'input', assert => {

    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    beforeTest( display );

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

    afterTest( display );    
  } );


  QUnit.test( 'change', assert => {

    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    beforeTest( display );

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

    afterTest( display );
  } );

  QUnit.test( 'keydown/keyup', assert => {

    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    beforeTest( display );

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

    afterTest( display );
  } );

  QUnit.test( 'Global KeyStateTracker tests', assert => {

    const rootNode = new Node( { tagName: 'div' } );
    const display = new Display( rootNode ); // eslint-disable-line
    beforeTest( display );

    const a = new Node( { tagName:'button' } );
    const b = new Node( { tagName:'button' } );
    const c = new Node( { tagName:'button' } );
    const d = new Node( { tagName:'button' } );

    a.addChild( b );
    b.addChild( c );
    c.addChild( d );
    rootNode.addChild( a );

    const dPrimarySibling = d.accessibleInstances[ 0 ].peer.primarySibling;
    triggerDOMEvent( 'keydown', dPrimarySibling, KeyboardUtil.KEY_RIGHT_ARROW );

    assert.ok( Display.keyStateTracker.isKeyDown( KeyboardUtil.KEY_RIGHT_ARROW ), 'global keyStateTracker should be updated with right arrow key down' );

    afterTest( display );
  } );
} );