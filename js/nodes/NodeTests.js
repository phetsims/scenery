// Copyright 2017-2020, University of Colorado Boulder

/**
 * Node tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import Property from '../../../axon/js/Property.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Range from '../../../dot/js/Range.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Shape from '../../../kite/js/Shape.js';
import HSlider from '../../../sun/js/HSlider.js';
import Tandem from '../../../tandem/js/Tandem.js';
import Display from '../display/Display.js';
import Touch from '../input/Touch.js';
import Node from './Node.js';
import Rectangle from './Rectangle.js';

QUnit.module( 'Node' );

function fakeTouchPointer( vector ) {
  return new Touch( 0, vector, {} );
}

QUnit.test( 'Mouse and Touch areas', function( assert ) {
  const node = new Node();
  const rect = new Rectangle( 0, 0, 100, 50 );
  rect.pickable = true;

  node.addChild( rect );

  assert.ok( !!rect.hitTest( new Vector2( 10, 10 ) ), 'Rectangle intersection' );
  assert.ok( !!rect.hitTest( new Vector2( 90, 10 ) ), 'Rectangle intersection' );
  assert.ok( !rect.hitTest( new Vector2( -10, 10 ) ), 'Rectangle no intersection' );

  node.touchArea = Shape.rectangle( -50, -50, 100, 100 );

  assert.ok( !!node.hitTest( new Vector2( 10, 10 ) ), 'Node intersection' );
  assert.ok( !!node.hitTest( new Vector2( 90, 10 ) ), 'Node intersection' );
  assert.ok( !node.hitTest( new Vector2( -10, 10 ) ), 'Node no intersection' );

  assert.ok( !!node.trailUnderPointer( fakeTouchPointer( new Vector2( 10, 10 ) ) ), 'Node intersection (isTouch)' );
  assert.ok( !!node.trailUnderPointer( fakeTouchPointer( new Vector2( 90, 10 ) ) ), 'Node intersection (isTouch)' );
  assert.ok( !!node.trailUnderPointer( fakeTouchPointer( new Vector2( -10, 10 ) ) ), 'Node intersection (isTouch)' );

  node.clipArea = Shape.rectangle( 0, 0, 50, 50 );

  // points outside the clip area shouldn't register as hits
  assert.ok( !!node.trailUnderPointer( fakeTouchPointer( new Vector2( 10, 10 ) ) ), 'Node intersection (isTouch with clipArea)' );
  assert.ok( !node.trailUnderPointer( fakeTouchPointer( new Vector2( 90, 10 ) ) ), 'Node no intersection (isTouch with clipArea)' );
  assert.ok( !node.trailUnderPointer( fakeTouchPointer( new Vector2( -10, 10 ) ) ), 'Node no intersection (isTouch with clipArea)' );
} );


const epsilon = 0.000000001;

QUnit.test( 'Points (parent and child)', function( assert ) {
  const a = new Node();
  const b = new Node();
  a.addChild( b );
  a.x = 10;
  b.y = 10;

  assert.ok( new Vector2( 5, 15 ).equalsEpsilon( b.localToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToParentPoint on child' );
  assert.ok( new Vector2( 15, 5 ).equalsEpsilon( a.localToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToParentPoint on root' );

  assert.ok( new Vector2( 5, -5 ).equalsEpsilon( b.parentToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToLocalPoint on child' );
  assert.ok( new Vector2( -5, 5 ).equalsEpsilon( a.parentToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToLocalPoint on root' );

  assert.ok( new Vector2( 15, 15 ).equalsEpsilon( b.localToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToGlobalPoint on child' );
  assert.ok( new Vector2( 15, 5 ).equalsEpsilon( a.localToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToGlobalPoint on root (same as localToparent)' );

  assert.ok( new Vector2( -5, -5 ).equalsEpsilon( b.globalToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToLocalPoint on child' );
  assert.ok( new Vector2( -5, 5 ).equalsEpsilon( a.globalToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToLocalPoint on root (same as localToparent)' );

  assert.ok( new Vector2( 15, 5 ).equalsEpsilon( b.parentToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToGlobalPoint on child' );
  assert.ok( new Vector2( 5, 5 ).equalsEpsilon( a.parentToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToGlobalPoint on root' );

  assert.ok( new Vector2( -5, 5 ).equalsEpsilon( b.globalToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToParentPoint on child' );
  assert.ok( new Vector2( 5, 5 ).equalsEpsilon( a.globalToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToParentPoint on root' );

} );

QUnit.test( 'Bounds (parent and child)', function( assert ) {
  const a = new Node();
  const b = new Node();
  a.addChild( b );
  a.x = 10;
  b.y = 10;

  const bounds = new Bounds2( 4, 4, 20, 30 );

  assert.ok( new Bounds2( 4, 14, 20, 40 ).equalsEpsilon( b.localToParentBounds( bounds ), epsilon ), 'localToParentBounds on child' );
  assert.ok( new Bounds2( 14, 4, 30, 30 ).equalsEpsilon( a.localToParentBounds( bounds ), epsilon ), 'localToParentBounds on root' );

  assert.ok( new Bounds2( 4, -6, 20, 20 ).equalsEpsilon( b.parentToLocalBounds( bounds ), epsilon ), 'parentToLocalBounds on child' );
  assert.ok( new Bounds2( -6, 4, 10, 30 ).equalsEpsilon( a.parentToLocalBounds( bounds ), epsilon ), 'parentToLocalBounds on root' );

  assert.ok( new Bounds2( 14, 14, 30, 40 ).equalsEpsilon( b.localToGlobalBounds( bounds ), epsilon ), 'localToGlobalBounds on child' );
  assert.ok( new Bounds2( 14, 4, 30, 30 ).equalsEpsilon( a.localToGlobalBounds( bounds ), epsilon ), 'localToGlobalBounds on root (same as localToParent)' );

  assert.ok( new Bounds2( -6, -6, 10, 20 ).equalsEpsilon( b.globalToLocalBounds( bounds ), epsilon ), 'globalToLocalBounds on child' );
  assert.ok( new Bounds2( -6, 4, 10, 30 ).equalsEpsilon( a.globalToLocalBounds( bounds ), epsilon ), 'globalToLocalBounds on root (same as localToParent)' );

  assert.ok( new Bounds2( 14, 4, 30, 30 ).equalsEpsilon( b.parentToGlobalBounds( bounds ), epsilon ), 'parentToGlobalBounds on child' );
  assert.ok( new Bounds2( 4, 4, 20, 30 ).equalsEpsilon( a.parentToGlobalBounds( bounds ), epsilon ), 'parentToGlobalBounds on root' );

  assert.ok( new Bounds2( -6, 4, 10, 30 ).equalsEpsilon( b.globalToParentBounds( bounds ), epsilon ), 'globalToParentBounds on child' );
  assert.ok( new Bounds2( 4, 4, 20, 30 ).equalsEpsilon( a.globalToParentBounds( bounds ), epsilon ), 'globalToParentBounds on root' );
} );

QUnit.test( 'Points (order of transforms)', function( assert ) {
  const a = new Node();
  const b = new Node();
  const c = new Node();
  a.addChild( b );
  b.addChild( c );
  a.x = 10;
  b.scale( 2 );
  c.y = 10;

  assert.ok( new Vector2( 20, 30 ).equalsEpsilon( c.localToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToGlobalPoint' );
  assert.ok( new Vector2( -2.5, -7.5 ).equalsEpsilon( c.globalToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToLocalPoint' );
  assert.ok( new Vector2( 20, 10 ).equalsEpsilon( c.parentToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToGlobalPoint' );
  assert.ok( new Vector2( -2.5, 2.5 ).equalsEpsilon( c.globalToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToParentPoint' );
} );

QUnit.test( 'Bounds (order of transforms)', function( assert ) {
  const a = new Node();
  const b = new Node();
  const c = new Node();
  a.addChild( b );
  b.addChild( c );
  a.x = 10;
  b.scale( 2 );
  c.y = 10;

  const bounds = new Bounds2( 4, 4, 20, 30 );

  assert.ok( new Bounds2( 18, 28, 50, 80 ).equalsEpsilon( c.localToGlobalBounds( bounds ), epsilon ), 'localToGlobalBounds' );
  assert.ok( new Bounds2( -3, -8, 5, 5 ).equalsEpsilon( c.globalToLocalBounds( bounds ), epsilon ), 'globalToLocalBounds' );
  assert.ok( new Bounds2( 18, 8, 50, 60 ).equalsEpsilon( c.parentToGlobalBounds( bounds ), epsilon ), 'parentToGlobalBounds' );
  assert.ok( new Bounds2( -3, 2, 5, 15 ).equalsEpsilon( c.globalToParentBounds( bounds ), epsilon ), 'globalToParentBounds' );
} );

QUnit.test( 'Trail and Node transform equivalence', function( assert ) {
  const a = new Node();
  const b = new Node();
  const c = new Node();
  a.addChild( b );
  b.addChild( c );
  a.x = 10;
  b.scale( 2 );
  c.y = 10;

  const trailMatrix = c.getUniqueTrail().getMatrix();
  const nodeMatrix = c.getUniqueTransform().getMatrix();
  assert.ok( trailMatrix.equalsEpsilon( nodeMatrix, epsilon ), 'Trail and Node transform equivalence' );
} );

QUnit.test( 'Node.enabledProperty', assert => {

  let node = new Node();

  testEnabledNode( assert, node, 'For Node' );

  const disabledOpacity = .2;
  node = new Node( {
    disabledOpacity: disabledOpacity
  } );

  assert.ok( node.opacity === new Node().opacity, 'opacity should default to Node default' );
  node.enabled = false;
  assert.ok( node.opacity === disabledOpacity, 'test disabled opacity' );

  node.dispose();

  // TinyProperty.isDisposed is only defined when assertions are enabled, for performance
  window.assert && assert.ok( node.enabledProperty.isDisposed, 'should be disposed' );

  const myEnabledProperty = new BooleanProperty( true );
  const defaultListenerCount = myEnabledProperty.changedEmitter.getListenerCount();
  const node2 = new Node( {
    enabledProperty: myEnabledProperty
  } );
  assert.ok( myEnabledProperty.changedEmitter.getListenerCount() > defaultListenerCount, 'listener count should be more since passing in enabledProperty' );

  node2.dispose();
  assert.ok( myEnabledProperty.changedEmitter.getListenerCount() === defaultListenerCount, 'listener count should match original' );
} );

QUnit.test( 'Node.enabledProperty with PDOM', assert => {

  const rootNode = new Node( { tagName: 'div' } );
  var display = new Display( rootNode ); // eslint-disable-line
  document.body.appendChild( display.domElement );

  const a11yNode = new Node( {
    tagName: 'p'
  } );

  rootNode.addChild( a11yNode );
  assert.ok( a11yNode.accessibleInstances.length === 1, 'should have an instance when attached to display' );
  assert.ok( !!a11yNode.accessibleInstances[ 0 ].peer, 'should have a peer' );

  // TODO: is it important that aria-disabled is false on all enabled Nodes? See https://github.com/phetsims/scenery/issues/1100
  // assert.ok( a11yNode.accessibleInstances[ 0 ].peer.primarySibling.getAttribute( 'aria-disabled' ) === 'false', 'should be enabled' );

  a11yNode.enabled = false;
  assert.ok( a11yNode.accessibleInstances[ 0 ].peer.primarySibling.getAttribute( 'aria-disabled' ) === 'true', 'should be enabled' );
  testEnabledNode( assert, a11yNode, 'For accessible Node' );
} );

QUnit.test( 'Node.enabledProperty in Slider', assert => {
  let slider = new HSlider( new Property( 0 ), new Range( 0, 10 ), {
    tandem: Tandem.GENERAL.createTandem( 'mySlider' )
  } );
  testEnabledNode( assert, slider, 'For Slider' );
  slider.dispose();

  const myEnabledProperty = new BooleanProperty( true, { tandem: Tandem.GENERAL.createTandem( 'myEnabledProperty' ) } );
  slider = new HSlider( new Property( 0 ), new Range( 0, 10 ), {
    tandem: Tandem.GENERAL.createTandem( 'mySlider' ),
    enabledProperty: myEnabledProperty
  } );
  testEnabledNode( assert, slider, 'For Slider' );
  slider.dispose();
  myEnabledProperty.dispose();
} );

/**
 * Test basic functionality for an object that mixes in EnabledComponent
 * @param {Object} assert - from QUnit
 * @param {Object} enabledNode - mixed in with EnabledComponent
 * @param {string} message - to tack onto assert messages
 */
function testEnabledNode( assert, enabledNode, message ) {
  assert.ok( enabledNode.enabledProperty instanceof Property || enabledNode.enabledProperty instanceof TinyProperty, `${message}: enabledProperty should exist` );

  assert.ok( enabledNode.enabledProperty.value === enabledNode.enabled, `${message}: test getter` );

  enabledNode.enabled = false;
  assert.ok( enabledNode.enabled === false, `${message}: test setter` );
  assert.ok( enabledNode.enabledProperty.value === enabledNode.enabled, `${message}: test getter after setting` );
  assert.ok( enabledNode.enabledProperty.value === false, `${message}: test getter after setting` );
}

if ( Tandem.PHET_IO_ENABLED ) {

  QUnit.test( 'Node instrumented visibleProperty', assert => testInstrumentedNodeProperty( assert, 'visible', 'visibleProperty', 'setVisibleProperty', true ) );

  QUnit.test( 'Node instrumented pickableProperty', assert => testInstrumentedNodeProperty( assert, 'pickable', 'pickableProperty', 'setPickableProperty', Node.DEFAULT_OPTIONS.pickablePropertyPhetioInstrumented ) );

  QUnit.test( 'Node instrumented enabledProperty', assert => testInstrumentedNodeProperty( assert, 'enabled', 'enabledProperty', 'setEnabledProperty', Node.DEFAULT_OPTIONS.enabledPropertyPhetioInstrumented ) );

  /**
   * Factor out a way to test added Properties to Node and their PhET-iO instrumentation
   * @param {Object} assert - from qunit test
   * @param {string} nodeField - name of getter/setter, like `visible`
   * @param {string} nodeProperty - name of public property, like `visibleProperty`
   * @param {string} nodePropertySetter - name of setter function, like `setVisibleProperty`
   * @param {boolean} ownedPropertyPhetioInstrumented - default value of *PhetioInstrumented option in Node.
   */
  const testInstrumentedNodeProperty = ( assert, nodeField, nodeProperty, nodePropertySetter, ownedPropertyPhetioInstrumented ) => {

    // TODO: Use the AuxiliaryTandemRegistry?  See https://github.com/phetsims/tandem/issues/187
    const wasLaunched = Tandem.launched;
    if ( !Tandem.launched ) {
      Tandem.launch();
    }

    const apiValidation = phet.tandem.phetioAPIValidation;
    const previousAPIValidationEnabled = apiValidation.enabled;
    const previousSimStarted = apiValidation.simHasStarted;

    apiValidation.simHasStarted = false;

    const testNodeAndProperty = ( node, property ) => {
      const initialValue = node[ nodeField ];
      assert.ok( property.value === node[ nodeField ], 'initial values should be the same' );
      node[ nodeField ] = !initialValue;
      assert.ok( property.value === !initialValue, 'property should reflect node change' );
      property.value = initialValue;
      assert.ok( node[ nodeField ] === initialValue, 'node should reflect property change' );

      node[ nodeField ] = initialValue;
    };

    const instrumentedProperty = new BooleanProperty( false, { tandem: Tandem.GENERAL.createTandem( `${nodeField}MyProperty` ) } );
    const otherInstrumentedProperty = new BooleanProperty( false, { tandem: Tandem.GENERAL.createTandem( `${nodeField}MyOtherProperty` ) } );
    const uninstrumentedProperty = new BooleanProperty( false );

    /***************************************
     /* Testing uninstrumented Nodes
     */


      // uninstrumentedNode => no property (before startup)
    let uninstrumented = new Node();
    assert.ok( uninstrumented[ nodeProperty ].targetProperty === undefined );
    testNodeAndProperty( uninstrumented, uninstrumented[ nodeProperty ] );

    // uninstrumentedNode => uninstrumented property (before startup)
    uninstrumented = new Node( { [ nodeProperty ]: uninstrumentedProperty } );
    assert.ok( uninstrumented[ nodeProperty ].targetProperty === uninstrumentedProperty );
    testNodeAndProperty( uninstrumented, uninstrumentedProperty );

    //uninstrumentedNode => instrumented property (before startup)
    uninstrumented = new Node();
    uninstrumented.mutate( {
      [ nodeProperty ]: instrumentedProperty
    } );
    assert.ok( uninstrumented[ nodeProperty ].targetProperty === instrumentedProperty );
    testNodeAndProperty( uninstrumented, instrumentedProperty );

    //  uninstrumentedNode => instrumented property => instrument the Node (before startup) OK
    uninstrumented = new Node();
    uninstrumented.mutate( {
      [ nodeProperty ]: instrumentedProperty
    } );
    uninstrumented.mutate( { tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` ) } );
    assert.ok( uninstrumented[ nodeProperty ].targetProperty === instrumentedProperty );
    testNodeAndProperty( uninstrumented, instrumentedProperty );
    uninstrumented.dispose();

    //////////////////////////////////////////////////
    apiValidation.simHasStarted = true;

    // uninstrumentedNode => no property (before startup)
    uninstrumented = new Node();
    assert.ok( uninstrumented[ nodeProperty ].targetProperty === undefined );
    testNodeAndProperty( uninstrumented, uninstrumented[ nodeProperty ] );

    // uninstrumentedNode => uninstrumented property (before startup)
    uninstrumented = new Node( { [ nodeProperty ]: uninstrumentedProperty } );
    assert.ok( uninstrumented[ nodeProperty ].targetProperty === uninstrumentedProperty );
    testNodeAndProperty( uninstrumented, uninstrumentedProperty );

    //uninstrumentedNode => instrumented property (before startup)
    uninstrumented = new Node();
    uninstrumented.mutate( {
      [ nodeProperty ]: instrumentedProperty
    } );
    assert.ok( uninstrumented[ nodeProperty ].targetProperty === instrumentedProperty );
    testNodeAndProperty( uninstrumented, instrumentedProperty );

    //  uninstrumentedNode => instrumented property => instrument the Node (before startup) OK
    uninstrumented = new Node();
    uninstrumented.mutate( {
      [ nodeProperty ]: instrumentedProperty
    } );

    uninstrumented.mutate( { tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` ) } );
    assert.ok( uninstrumented[ nodeProperty ].targetProperty === instrumentedProperty );
    testNodeAndProperty( uninstrumented, instrumentedProperty );
    uninstrumented.dispose();
    apiValidation.simHasStarted = false;


    /***************************************
     /* Testing instrumented nodes
     */

      // instrumentedNodeWithDefaultInstrumentedProperty => instrumented property (before startup)
    let instrumented = new Node( {
        tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` ),
        [ `${nodeProperty}PhetioInstrumented` ]: true
      } );
    assert.ok( instrumented[ nodeProperty ].targetProperty === instrumented[ nodeProperty ].ownedPhetioProperty );
    assert.ok( instrumented.linkedElements.length === 0, `no linked elements for default ${nodeProperty}` );
    testNodeAndProperty( instrumented, instrumented[ nodeProperty ] );
    instrumented.dispose();

    // instrumentedNodeWithDefaultInstrumentedProperty => uninstrumented property (before startup)
    instrumented = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` ),
      [ `${nodeProperty}PhetioInstrumented` ]: true
    } );
    window.assert && assert.throws( () => {
      instrumented.mutate( { [ nodeProperty ]: uninstrumentedProperty } );
    }, `cannot remove instrumentation from the Node's ${nodeProperty}` );
    instrumented.dispose();

    // instrumentedNodeWithPassedInInstrumentedProperty => instrumented property (before startup)
    instrumented = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` ),
      [ `${nodeProperty}PhetioInstrumented` ]: true
    } );
    instrumented.mutate( { [ nodeProperty ]: instrumentedProperty } );
    assert.ok( instrumented[ nodeProperty ].targetProperty === instrumentedProperty );
    assert.ok( instrumented.linkedElements.length === 1, 'added linked element' );
    assert.ok( instrumented.linkedElements[ 0 ].element === instrumentedProperty,
      `added linked element should be for ${nodeProperty}` );
    testNodeAndProperty( instrumented, instrumentedProperty );
    instrumented.dispose();

    instrumented = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` ),
      [ nodeProperty ]: instrumentedProperty
    } );
    assert.ok( instrumented[ nodeProperty ].targetProperty === instrumentedProperty );
    assert.ok( instrumented.linkedElements.length === 1, 'added linked element' );
    assert.ok( instrumented.linkedElements[ 0 ].element === instrumentedProperty,
      `added linked element should be for ${nodeProperty}` );
    testNodeAndProperty( instrumented, instrumentedProperty );
    instrumented.dispose();

    // instrumentedNodeWithPassedInInstrumentedProperty => uninstrumented property (before startup)
    instrumented = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` ),
      [ nodeProperty ]: instrumentedProperty
    } );
    window.assert && assert.throws( () => {
      instrumented.mutate( { [ nodeProperty ]: uninstrumentedProperty } );
    }, `cannot remove instrumentation from the Node's ${nodeProperty}` );
    instrumented.dispose();
    instrumented = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` )
    } );
    instrumented.mutate( { [ nodeProperty ]: instrumentedProperty } );
    window.assert && assert.throws( () => {
      instrumented.mutate( { [ nodeProperty ]: uninstrumentedProperty } );
    }, `cannot remove instrumentation from the Node's ${nodeProperty}` );
    instrumented.dispose();

    apiValidation.enabled = true;
    apiValidation.simHasStarted = true;
    // instrumentedNodeWithDefaultInstrumentedProperty => instrumented property (after startup)
    const instrumented1 = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyUniquelyNamedNodeThatWillNotBeDuplicated1` ),
      [ `${nodeProperty}PhetioInstrumented` ]: true
    } );
    assert.ok( instrumented1[ nodeProperty ].targetProperty === instrumented1[ nodeProperty ].ownedPhetioProperty );
    assert.ok( instrumented1.linkedElements.length === 0, `no linked elements for default ${nodeProperty}` );
    testNodeAndProperty( instrumented1, instrumented1[ nodeProperty ] );

    // instrumentedNodeWithDefaultInstrumentedProperty => uninstrumented property (after startup)
    const instrumented2 = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyUniquelyNamedNodeThatWillNotBeDuplicated2` ),
      [ `${nodeProperty}PhetioInstrumented` ]: true
    } );
    window.assert && assert.throws( () => {
      instrumented2[ nodePropertySetter ]( uninstrumentedProperty );
    }, `cannot remove instrumentation from the Node's ${nodeProperty}` );

    // instrumentedNodeWithPassedInInstrumentedProperty => instrumented property (after startup)
    const instrumented3 = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyUniquelyNamedNodeThatWillNotBeDuplicated3` ),
      [ nodeProperty ]: instrumentedProperty
    } );

    window.assert && assert.throws( () => {
      instrumented3.mutate( { [ nodeProperty ]: otherInstrumentedProperty } );
    }, 'cannot swap out one instrumented for another' );

    // instrumentedNodeWithPassedInInstrumentedProperty => uninstrumented property (after startup)
    const instrumented4 = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyUniquelyNamedNodeThatWillNotBeDuplicated4` ),
      [ nodeProperty ]: instrumentedProperty
    } );
    window.assert && assert.throws( () => {
      instrumented4.mutate( { [ nodeProperty ]: uninstrumentedProperty } );
    }, `cannot remove instrumentation from the Node's ${nodeProperty}` );
    const instrumented5 = new Node( {} );
    instrumented5.mutate( { [ nodeProperty ]: instrumentedProperty } );
    instrumented5.mutate( { tandem: Tandem.GENERAL.createTandem( `${nodeField}MyUniquelyNamedNodeThatWillNotBeDuplicated5` ) } );
    window.assert && assert.throws( () => {
      instrumented5.mutate( { [ nodeProperty ]: uninstrumentedProperty } );
    }, `cannot remove instrumentation from the Node's ${nodeProperty}` );
    apiValidation.enabled = false;

    instrumented1.dispose();

    // These can't be disposed because they were broken while creating (on purpose in an assert.throws()). These elements
    // have special Tandem component names to make sure that they don't interfere with other tests (since they can't be
    // removed from the registry
    // instrumented2.dispose();
    // instrumented3.dispose();
    // instrumented4.dispose();
    // instrumented5.dispose();

    instrumented = new Node( {
      tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` ),
      [ `${nodeProperty}PhetioInstrumented` ]: true
    } );
    window.assert && assert.throws( () => {
      instrumented[ nodePropertySetter ]( null );
    }, `cannot clear out an instrumented ${nodeProperty}` );
    instrumented.dispose();


    // If by default this property isn't instrumented, then this should cause an error
    if ( !ownedPropertyPhetioInstrumented ) {

      instrumented = new Node( {
        tandem: Tandem.GENERAL.createTandem( `${nodeField}MyNode` )
      } );
      window.assert && assert.throws( () => {
        instrumented[ `${nodeProperty}PhetioInstrumented` ] = true;
      }, `cannot set ${nodeProperty}PhetioInstrumented after instrumentation` );
      instrumented.dispose();
    }


    instrumentedProperty.dispose();
    otherInstrumentedProperty.dispose();
    apiValidation.simHasStarted = previousSimStarted;
    apiValidation.enabled = previousAPIValidationEnabled;

    if ( !wasLaunched ) {
      Tandem.unlaunch();
    }
  };
}
