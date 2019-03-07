// Copyright 2017, University of Colorado Boulder

/**
 * DragListener tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var Bounds2 = require( 'DOT/Bounds2' );
  var DragListener = require( 'SCENERY/listeners/DragListener' );
  var ListenerTestUtils = require( 'SCENERY/listeners/ListenerTestUtils' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Property = require( 'AXON/Property' );
  var Transform3 = require( 'DOT/Transform3' );
  var Vector2 = require( 'DOT/Vector2' );
  var Vector2Property = require( 'DOT/Vector2Property' );

  QUnit.module( 'DragListener' );

  QUnit.test( 'translateNode', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      var listener = new DragListener( {
        translateNode: true
      } );
      rect.addInputListener( listener );

      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseDown( display, 10, 10 );
      ListenerTestUtils.mouseMove( display, 20, 15 );
      ListenerTestUtils.mouseUp( display, 20, 15 );
      assert.equal( rect.x, 10, 'Drag with translateNode should have changed the x translation' );
      assert.equal( rect.y, 5, 'Drag with translateNode should have changed the y translation' );
    } );
  } );

  QUnit.test( 'translateNode with applyOffset:false', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      var listener = new DragListener( {
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
    } );
  } );

  QUnit.test( 'translateNode with trackAncestors', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      var listener = new DragListener( {
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
    } );
  } );

  QUnit.test( 'locationProperty with hooks', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      var locationProperty = new Vector2Property( Vector2.ZERO );
      locationProperty.linkAttribute( rect, 'translation' );

      var listener = new DragListener( {
        locationProperty: locationProperty
      } );
      rect.addInputListener( listener );

      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseDown( display, 10, 10 );
      ListenerTestUtils.mouseMove( display, 20, 15 );
      ListenerTestUtils.mouseUp( display, 20, 15 );
      assert.equal( locationProperty.value.x, 10, 'Drag with translateNode should have changed the x translation' );
      assert.equal( locationProperty.value.y, 5, 'Drag with translateNode should have changed the y translation' );
    } );
  } );

  QUnit.test( 'locationProperty with hooks and transform', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      var locationProperty = new Vector2Property( Vector2.ZERO );
      var transform = new Transform3( Matrix3.translation( 5, 3 ).timesMatrix( Matrix3.scale( 2 ) ).timesMatrix( Matrix3.rotation2( Math.PI / 4 ) ) );

      // Starts at 5,3
      locationProperty.link( function( location ) {
        rect.translation = transform.transformPosition2( location );
      } );

      var listener = new DragListener( {
        locationProperty: locationProperty,
        transform: transform
      } );
      rect.addInputListener( listener );

      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseDown( display, 10, 10 );
      ListenerTestUtils.mouseMove( display, 20, 15 );
      ListenerTestUtils.mouseUp( display, 20, 15 );
      assert.equal( rect.x, 15, '[x] Started at 5, moved by 10' );
      assert.equal( rect.y, 8, '[y] Started at 3, moved by 5' );
    } );
  } );

  QUnit.test( 'locationProperty with dragBounds', function( assert ) {
    ListenerTestUtils.simpleRectangleTest( function( display, rect, node ) {
      var locationProperty = new Vector2Property( Vector2.ZERO );

      locationProperty.link( function( location ) {
        rect.translation = location;
      } );

      var listener = new DragListener( {
        locationProperty: locationProperty,
        dragBoundsProperty: new Property( new Bounds2( 0, 0, 5, 5 ) )
      } );
      rect.addInputListener( listener );

      ListenerTestUtils.mouseMove( display, 10, 10 );
      ListenerTestUtils.mouseDown( display, 10, 10 );
      ListenerTestUtils.mouseMove( display, 50, 30 );
      ListenerTestUtils.mouseUp( display, 50, 30 );
      assert.equal( locationProperty.value.x, 5, '[x] Should be limited to 5 by dragBounds' );
      assert.equal( locationProperty.value.y, 5, '[y] Should be limited to 5 by dragBounds  ' );
    } );
  } );
} );
