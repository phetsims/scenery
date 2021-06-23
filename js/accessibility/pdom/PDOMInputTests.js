// Copyright 2018-2021, University of Colorado Boulder

/**
 * Tests related to ParallelDOM input and events.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import merge from '../../../../phet-core/js/merge.js';
import Display from '../../display/Display.js';
import Node from '../../nodes/Node.js';
import Rectangle from '../../nodes/Rectangle.js';
import globalKeyStateTracker from '../globalKeyStateTracker.js';
import KeyboardUtils from '../KeyboardUtils.js';

// constants
const TEST_LABEL = 'Test Label';
const TEST_LABEL_2 = 'Test Label 2';

QUnit.module( 'PDOMInput' );

/**
 * Set up a test for accessible input by attaching a root node to a display and initializing events.
 * @param {Display} display
 */
const beforeTest = display => {
  display.initializeEvents();
  document.body.appendChild( display.domElement );
};

/**
 * Clean up a test by detaching events and removing the element from the DOM so that it doesn't interfere
 * with QUnit UI.
 * @param {Display} display
 */
const afterTest = display => {
  document.body.removeChild( display.domElement );
  display.dispose();
};

const dispatchEvent = ( domElement, event ) => {
  const Constructor = event.startsWith( 'key' ) ? window.KeyboardEvent : window.Event;
  domElement.dispatchEvent( new Constructor( event, {
    bubbles: true, // that is vital to all that scenery events hold near and dear to their hearts.
    key: KeyboardUtils.KEY_TAB
  } ) );
};

// create a fake DOM event and delegate to an HTMLElement
// TODO: Can this replace the dispatchEvent function above? EXTRA_TODO use KeyboardFuzzer.triggerDOMEvent as a guide to rewrite this.
const triggerDOMEvent = ( event, element, key, options ) => {

  options = merge( {

    // secondary target for the event, behavior depends on event type
    relatedTarget: null
  }, options );

  const eventObj = document.createEventObject ?
                   document.createEventObject() : document.createEvent( 'Events' );

  if ( eventObj.initEvent ) {
    eventObj.initEvent( event, true, true );
  }

  eventObj.code = key;
  eventObj.relatedTarget = options.relatedTarget;

  element.dispatchEvent ? element.dispatchEvent( eventObj ) : element.fireEvent( `on${event}`, eventObj );
};

QUnit.test( 'focusin/focusout (focus/blur)', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode ); // eslint-disable-line
  beforeTest( display );

  const a = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );
  const b = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );
  const c = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

  // rootNode
  //   /  \
  //  a    b
  //        \
  //         c
  rootNode.addChild( a );
  rootNode.addChild( b );
  b.addChild( c );

  let aGotFocus = false;
  let aLostFocus = false;
  let bGotFocus = false;
  let bGotBlur = false;
  let bGotFocusIn = false;
  let bGotFocusOut = false;
  let cGotFocusIn = false;
  let cGotFocusOut = false;

  const resetFocusVariables = () => {
    aGotFocus = false;
    aLostFocus = false;
    bGotFocus = false;
    bGotBlur = false;
    bGotFocusIn = false;
    bGotFocusOut = false;
    cGotFocusIn = false;
    cGotFocusOut = false;
  };

  a.addInputListener( {
    focus() {
      aGotFocus = true;
    },
    blur() {
      aLostFocus = true;
    }
  } );

  b.addInputListener( {
    focus() {
      bGotFocus = true;
    },
    blur() {
      bGotBlur = true;
    },
    focusin() {
      bGotFocusIn = true;
    },
    focusout() {
      bGotFocusOut = true;
    }
  } );

  c.addInputListener( {
    focusin() {
      cGotFocusIn = true;
    },
    focusout() {
      cGotFocusOut = true;
    }
  } );

  a.focus();

  assert.ok( aGotFocus, 'a should have been focused' );
  assert.ok( !aLostFocus, 'a should not blur' );
  resetFocusVariables();

  b.focus();
  assert.ok( bGotFocus, 'b should have been focused' );
  assert.ok( aLostFocus, 'a should have lost focused' );
  resetFocusVariables();

  c.focus();
  assert.ok( !bGotFocus, 'b should not receive focus (doesnt bubble)' );
  assert.ok( cGotFocusIn, 'c should receive a focusin' );
  assert.ok( bGotFocusIn, 'b should receive a focusin (from bubbling)' );
  resetFocusVariables();

  c.blur();
  assert.ok( !bGotBlur, 'b should not receive a blur event (doesnt bubble)' );
  assert.ok( cGotFocusOut, 'c should have received a focusout' );
  assert.ok( bGotFocusOut, 'c should have received a focusout (from bubbling)' );

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

  const aPrimarySibling = buttonA.pdomInstances[ 0 ].peer.primarySibling;
  const bPrimarySibling = buttonB.pdomInstances[ 0 ].peer.primarySibling;

  // test that a blur listener on a node overides the "tab" like navigation moving focus to the next element
  buttonA.focus();
  assert.ok( buttonA.focused, 'butonA has focus initially' );

  const overrideFocusListener = {
    blur: event => {
      buttonC.focus();
    }
  };
  buttonA.addInputListener( overrideFocusListener );

  // mimic a "tab" interaction, attempting to move focus to the next element
  triggerDOMEvent( 'focusout', aPrimarySibling, KeyboardUtils.KEY_TAB, {
    relatedTarget: bPrimarySibling
  } );

  // the blur listener on buttonA should override the movement of focus on "tab" like interaction
  assert.ok( buttonC.focused, 'butonC now has focus' );

  // test that a blur listener can prevent focus from moving to another element after "tab" like navigation
  buttonA.removeInputListener( overrideFocusListener );
  buttonA.focus();
  const makeUnfocusableListener = {
    blur: event => {
      buttonB.focusable = false;
    }
  };
  buttonA.addInputListener( makeUnfocusableListener );
  triggerDOMEvent( 'focusout', aPrimarySibling, KeyboardUtils.KEY_TAB, {
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


  a.pdomInstances[ 0 ].peer.primarySibling.focus();
  assert.ok( gotFocus && !gotClick, 'focus first' );
  a.pdomInstances[ 0 ].peer.primarySibling.click(); // this works because it's a button
  assert.ok( gotClick && gotFocus && aClickCounter === 1, 'a should have been clicked' );

  let bClickCounter = 0;

  const b = new Rectangle( 0, 0, 20, 20, { tagName: 'button' } );

  b.addInputListener( {
    click() {
      bClickCounter++;
    }
  } );

  a.addChild( b );

  b.pdomInstances[ 0 ].peer.primarySibling.focus();
  b.pdomInstances[ 0 ].peer.primarySibling.click();
  assert.ok( bClickCounter === 1 && aClickCounter === 2, 'a should have been clicked with b' );
  a.pdomInstances[ 0 ].peer.primarySibling.click();
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
  a1.pdomInstances[ 0 ].peer.primarySibling.click();
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

  e.pdomInstances[ 0 ].peer.primarySibling.click();

  assert.ok( cClickCount === dClickCount && cClickCount === eClickCount && cClickCount === 1,
    'click should have bubbled to all parents' );

  d.pdomInstances[ 0 ].peer.primarySibling.click();


  assert.ok( cClickCount === 2 && dClickCount === 2 && eClickCount === 1,
    'd should not trigger click on e' );
  c.pdomInstances[ 0 ].peer.primarySibling.click();


  assert.ok( cClickCount === 3 && dClickCount === 2 && eClickCount === 1,
    'c should not trigger click on d or e' );

  // reset click count
  cClickCount = 0;
  dClickCount = 0;
  eClickCount = 0;

  c.pdomOrder = [ d, e ];

  e.pdomInstances[ 0 ].peer.primarySibling.click();
  assert.ok( cClickCount === 1 && dClickCount === 0 && eClickCount === 1,
    'pdomOrder means click should bypass d' );

  c.pdomInstances[ 0 ].peer.primarySibling.click();
  assert.ok( cClickCount === 2 && dClickCount === 0 && eClickCount === 1,
    'click c should not effect e or d.' );

  d.pdomInstances[ 0 ].peer.primarySibling.click();
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

  d.pdomOrder = [ f ];

  /* accessible instance tree:
       c
      / \
      d  e
      |
      f
  */

  f.pdomInstances[ 0 ].peer.primarySibling.click();
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
  const listener = { click: () => { a1.labelContent = TEST_LABEL; } };
  a1.addInputListener( listener );
  assert.ok( a1.inputListeners.length === 1, 'accessible listener added' );

  // verify added with hasInputListener
  assert.ok( a1.hasInputListener( listener ) === true, 'found with hasInputListener' );

  // fire the event
  a1.pdomInstances[ 0 ].peer.primarySibling.click();
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
  a1.pdomInstances[ 0 ].peer.primarySibling.click();
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

  a.pdomInstances[ 0 ].peer.primarySibling.focus();
  assert.ok( gotFocus && !gotInput, 'focus first' );

  dispatchEvent( a.pdomInstances[ 0 ].peer.primarySibling, 'input' );

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

  a.pdomInstances[ 0 ].peer.primarySibling.focus();
  assert.ok( gotFocus && !gotChange, 'focus first' );

  dispatchEvent( a.pdomInstances[ 0 ].peer.primarySibling, 'change' );

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

  a.pdomInstances[ 0 ].peer.primarySibling.focus();
  assert.ok( gotFocus && !gotKeydown, 'focus first' );

  dispatchEvent( a.pdomInstances[ 0 ].peer.primarySibling, 'keydown' );

  assert.ok( gotKeydown && gotFocus, 'a should have had keydown' );

  dispatchEvent( a.pdomInstances[ 0 ].peer.primarySibling, 'keyup' );
  assert.ok( gotKeydown && gotKeyup && gotFocus, 'a should have had keyup' );

  afterTest( display );
} );

QUnit.test( 'Global KeyStateTracker tests', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  const display = new Display( rootNode ); // eslint-disable-line
  beforeTest( display );

  const a = new Node( { tagName: 'button' } );
  const b = new Node( { tagName: 'button' } );
  const c = new Node( { tagName: 'button' } );
  const d = new Node( { tagName: 'button' } );

  a.addChild( b );
  b.addChild( c );
  c.addChild( d );
  rootNode.addChild( a );

  const dPrimarySibling = d.pdomInstances[ 0 ].peer.primarySibling;
  triggerDOMEvent( 'keydown', dPrimarySibling, KeyboardUtils.KEY_RIGHT_ARROW );

  assert.ok( globalKeyStateTracker.isKeyDown( KeyboardUtils.KEY_RIGHT_ARROW ), 'global keyStateTracker should be updated with right arrow key down' );

  afterTest( display );
} );