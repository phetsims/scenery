// Copyright 2017, University of Colorado Boulder

/**
 * Miscellaneous tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  QUnit.module( 'Miscellaneous' );

  var includeBleedingEdgeCanvasTests = false;

  QUnit.test( 'ES5 Object.defineProperty get/set', function( assert ) {
    var ob = { _key: 5 };
    Object.defineProperty( ob, 'key', {
      enumerable: true,
      configurable: true,
      get: function() { return this._key; },
      set: function( val ) { this._key = val; }
    } );
    ob.key += 1;
    assert.equal( ob._key, 6, 'incremented object value' );
  } );

  // QUnit.test( 'Canvas WebGL Context and Features', function(assert) {
  //   var canvas = document.createElement( 'canvas' );
  //   var context = canvas.getContext( "webgl" ) || canvas.getContext( "experimental-webgl" );
  //   assert.ok( context, 'context' );
  // } );

  if ( includeBleedingEdgeCanvasTests ) {
    // v5 canvas additions
    QUnit.module( 'Bleeding Edge Canvas Support' );

    QUnit.test( 'Canvas 2D v5 Features', function( assert ) {
      var canvas = document.createElement( 'canvas' );
      var context = canvas.getContext( '2d' );

      var neededMethods = [
        'addHitRegion',
        'ellipse',
        'resetClip',
        'resetTransform'
      ];
      _.each( neededMethods, function( method ) {
        assert.ok( context[ method ] !== undefined, 'context.' + method );
      } );
    } );

    QUnit.test( 'Path object support', function( assert ) {
      new Path( null ); // eslint-disable-line
    } );

    QUnit.test( 'Text width measurement in canvas', function( assert ) {
      var canvas = document.createElement( 'canvas' );
      var context = canvas.getContext( '2d' );
      var metrics = context.measureText( 'Hello World' );
      _.each( [ 'actualBoundingBoxLeft', 'actualBoundingBoxRight', 'actualBoundingBoxAscent', 'actualBoundingBoxDescent' ], function( method ) {
        assert.ok( metrics[ method ] !== undefined, 'metrics.' + method );
      } );
    } );
  }
} );