// Copyright 2013-2022, University of Colorado Boulder

/**
 * Miscellaneous tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

QUnit.module( 'Miscellaneous' );

const includeBleedingEdgeCanvasTests = false;

QUnit.test( 'ES5 Object.defineProperty get/set', assert => {
  const ob = { _key: 5 };
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

  QUnit.test( 'Canvas 2D v5 Features', assert => {
    const canvas = document.createElement( 'canvas' );
    const context = canvas.getContext( '2d' );

    const neededMethods = [
      'addHitRegion',
      'ellipse',
      'resetClip',
      'resetTransform'
    ];
    _.each( neededMethods, method => {
      assert.ok( context[ method ] !== undefined, `context.${method}` );
    } );
  } );

  QUnit.test( 'Path object support', assert => {
    new Path( null ); // eslint-disable-line no-new, no-undef
  } );

  QUnit.test( 'Text width measurement in canvas', assert => {
    const canvas = document.createElement( 'canvas' );
    const context = canvas.getContext( '2d' );
    const metrics = context.measureText( 'Hello World' );
    _.each( [ 'actualBoundingBoxLeft', 'actualBoundingBoxRight', 'actualBoundingBoxAscent', 'actualBoundingBoxDescent' ], method => {
      assert.ok( metrics[ method ] !== undefined, `metrics.${method}` );
    } );
  } );
}