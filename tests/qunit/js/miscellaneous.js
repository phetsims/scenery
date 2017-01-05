// Copyright 2002-2014, University of Colorado Boulder

(function() {
  'use strict';

  var includeBleedingEdgeCanvasTests = false;

  module( 'Scenery: Miscellaneous' );

  test( 'ES5 Object.defineProperty get/set', function() {
    var ob = { _key: 5 };
    Object.defineProperty( ob, 'key', {
      enumerable: true,
      configurable: true,
      get: function() { return this._key; },
      set: function( val ) { this._key = val; }
    } );
    ob.key += 1;
    equal( ob._key, 6, 'incremented object value' );
  } );

  // test( 'Canvas WebGL Context and Features', function() {
  //   var canvas = document.createElement( 'canvas' );
  //   var context = canvas.getContext( "webgl" ) || canvas.getContext( "experimental-webgl" );
  //   ok( context, 'context' );
  // } );

  if ( includeBleedingEdgeCanvasTests ) {
    // v5 canvas additions
    module( 'Bleeding Edge Canvas Support' );

    test( 'Canvas 2D v5 Features', function() {
      var canvas = document.createElement( 'canvas' );
      var context = canvas.getContext( '2d' );

      var neededMethods = [
        'addHitRegion',
        'ellipse',
        'resetClip',
        'resetTransform'
      ];
      _.each( neededMethods, function( method ) {
        ok( context[ method ] !== undefined, 'context.' + method );
      } );
    } );

    test( 'Path object support', function() {
      new scenery.Path( null ); // eslint-disable-line
    } );

    test( 'Text width measurement in canvas', function() {
      var canvas = document.createElement( 'canvas' );
      var context = canvas.getContext( '2d' );
      var metrics = context.measureText( 'Hello World' );
      _.each( [ 'actualBoundingBoxLeft', 'actualBoundingBoxRight', 'actualBoundingBoxAscent', 'actualBoundingBoxDescent' ], function( method ) {
        ok( metrics[ method ] !== undefined, 'metrics.' + method );
      } );
    } );
  }
})();
