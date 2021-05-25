// Copyright 2018-2021, University of Colorado Boulder

/**
 * DragListener tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Utils from '../../../dot/js/Utils.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Vector2Property from '../../../dot/js/Vector2Property.js';
import Tandem from '../../../tandem/js/Tandem.js';
import DragListener from './DragListener.js';
import ListenerTestUtils from './ListenerTestUtils.js';

QUnit.module( 'DragListener' );

QUnit.test( 'translateNode', assert => {
  ListenerTestUtils.simpleRectangleTest( ( display, rect, node ) => {
    const listener = new DragListener( {
      tandem: Tandem.ROOT_TEST.createTandem( 'myListener' ),
      translateNode: true
    } );
    rect.addInputListener( listener );

    ListenerTestUtils.mouseMove( display, 10, 10 );
    ListenerTestUtils.mouseDown( display, 10, 10 );
    ListenerTestUtils.mouseMove( display, 20, 15 );
    ListenerTestUtils.mouseUp( display, 20, 15 );
    assert.equal( rect.x, 10, 'Drag with translateNode should have changed the x translation' );
    assert.equal( rect.y, 5, 'Drag with translateNode should have changed the y translation' );
    listener.dispose();
  } );
} );

QUnit.test( 'translateNode with applyOffset:false', assert => {
  ListenerTestUtils.simpleRectangleTest( ( display, rect, node ) => {
    const listener = new DragListener( {
      tandem: Tandem.ROOT_TEST.createTandem( 'myListener' ),
      translateNode: true,
      applyOffset: false
    } );
    rect.addInputListener( listener );

    ListenerTestUtils.mouseMove( display, 10, 10 );
    ListenerTestUtils.mouseDown( display, 10, 10 );
    ListenerTestUtils.mouseMove( display, 20, 15 );
    ListenerTestUtils.mouseUp( display, 20, 15 );
    assert.equal( rect.x, 20, 'Drag should place the rect with its origin at the last mouse position (x)' );
    assert.equal( rect.y, 15, 'Drag should place the rect with its origin at the last mouse position (y)' );
    listener.dispose();
  } );
} );

QUnit.test( 'translateNode with trackAncestors', assert => {
  ListenerTestUtils.simpleRectangleTest( ( display, rect, node ) => {
    const listener = new DragListener( {
      tandem: Tandem.ROOT_TEST.createTandem( 'myListener' ),
      translateNode: true,
      trackAncestors: true
    } );
    rect.addInputListener( listener );

    ListenerTestUtils.mouseMove( display, 10, 10 );
    ListenerTestUtils.mouseDown( display, 10, 10 );
    node.x = 5;
    ListenerTestUtils.mouseMove( display, 20, 15 );
    ListenerTestUtils.mouseUp( display, 20, 15 );
    assert.equal( rect.x, 5, 'The x shift of 10 on the base node will have wiped out half of the drag change' );
    assert.equal( rect.y, 5, 'No y movement occurred of the base node' );
    listener.dispose();
  } );
} );

QUnit.test( 'positionProperty with hooks', assert => {
  ListenerTestUtils.simpleRectangleTest( ( display, rect, node ) => {
    const positionProperty = new Vector2Property( Vector2.ZERO );
    positionProperty.linkAttribute( rect, 'translation' );

    const listener = new DragListener( {
      tandem: Tandem.ROOT_TEST.createTandem( 'myListener' ),
      positionProperty: positionProperty
    } );
    rect.addInputListener( listener );

    ListenerTestUtils.mouseMove( display, 10, 10 );
    ListenerTestUtils.mouseDown( display, 10, 10 );
    ListenerTestUtils.mouseMove( display, 20, 15 );
    ListenerTestUtils.mouseUp( display, 20, 15 );
    assert.equal( positionProperty.value.x, 10, 'Drag with translateNode should have changed the x translation' );
    assert.equal( positionProperty.value.y, 5, 'Drag with translateNode should have changed the y translation' );
    listener.dispose();
  } );
} );

QUnit.test( 'positionProperty with hooks and transform', assert => {
  ListenerTestUtils.simpleRectangleTest( ( display, rect, node ) => {
    const positionProperty = new Vector2Property( Vector2.ZERO );
    const transform = new Transform3( Matrix3.translation( 5, 3 ).timesMatrix( Matrix3.scale( 2 ) ).timesMatrix( Matrix3.rotation2( Math.PI / 4 ) ) );

    // Starts at 5,3
    positionProperty.link( position => {
      rect.translation = transform.transformPosition2( position );
    } );

    const listener = new DragListener( {
      tandem: Tandem.ROOT_TEST.createTandem( 'myListener' ),
      positionProperty: positionProperty,
      transform: transform
    } );
    rect.addInputListener( listener );

    ListenerTestUtils.mouseMove( display, 10, 10 );
    ListenerTestUtils.mouseDown( display, 10, 10 );
    ListenerTestUtils.mouseMove( display, 20, 15 );
    ListenerTestUtils.mouseUp( display, 20, 15 );
    assert.equal( Utils.roundSymmetric( rect.x ), 15, '[x] Started at 5, moved by 10' );
    assert.equal( Utils.roundSymmetric( rect.y ), 8, '[y] Started at 3, moved by 5' );
    listener.dispose();
  } );
} );

QUnit.test( 'positionProperty with dragBounds', assert => {
  ListenerTestUtils.simpleRectangleTest( ( display, rect, node ) => {
    const positionProperty = new Vector2Property( Vector2.ZERO );

    positionProperty.link( position => {
      rect.translation = position;
    } );

    const listener = new DragListener( {
      tandem: Tandem.ROOT_TEST.createTandem( 'myListener' ),
      positionProperty: positionProperty,
      dragBoundsProperty: new Property( new Bounds2( 0, 0, 5, 5 ) )
    } );
    rect.addInputListener( listener );

    ListenerTestUtils.mouseMove( display, 10, 10 );
    ListenerTestUtils.mouseDown( display, 10, 10 );
    ListenerTestUtils.mouseMove( display, 50, 30 );
    ListenerTestUtils.mouseUp( display, 50, 30 );
    assert.equal( positionProperty.value.x, 5, '[x] Should be limited to 5 by dragBounds' );
    assert.equal( positionProperty.value.y, 5, '[y] Should be limited to 5 by dragBounds  ' );
    listener.dispose();
  } );
} );