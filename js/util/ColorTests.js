// Copyright 2017, University of Colorado Boulder

/**
 * Color tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var Color = require( 'SCENERY/util/Color' );

  QUnit.module( 'Color' );

  QUnit.test( 'RGB Hex', function( assert ) {
    var ff00cc = new Color( '#ff00cc' );
    assert.equal( ff00cc.r, 0xff, 'ff00cc red' );
    assert.equal( ff00cc.g, 0, 'ff00cc green' );
    assert.equal( ff00cc.b, 0xcc, 'ff00cc blue' );
    assert.equal( ff00cc.a, 1, 'ff00cc alpha' );
    assert.equal( ff00cc.toCSS(), 'rgb(255,0,204)', 'ff00cc css' );

    var f0c = new Color( '#f0c' );
    assert.equal( f0c.r, 0xff, 'f0c red' );
    assert.equal( f0c.g, 0, 'f0c green' );
    assert.equal( f0c.b, 0xcc, 'f0c blue' );
    assert.equal( f0c.a, 1, 'f0c alpha' );
    assert.equal( f0c.toCSS(), 'rgb(255,0,204)', 'f0c css' );
  } );

  QUnit.test( 'RGB Hex direct', function( assert ) {
    var ff00cc = new Color( 0xff00cc );
    assert.equal( ff00cc.r, 0xff, 'ff00cc red' );
    assert.equal( ff00cc.g, 0, 'ff00cc green' );
    assert.equal( ff00cc.b, 0xcc, 'ff00cc blue' );
    assert.equal( ff00cc.a, 1, 'ff00cc alpha' );
    assert.equal( ff00cc.toCSS(), 'rgb(255,0,204)', 'ff00cc css' );

    var ff00ccHalf = new Color( 0xff00cc, 0.5 );
    assert.equal( ff00ccHalf.r, 0xff, 'ff00ccHalf red' );
    assert.equal( ff00ccHalf.g, 0, 'ff00ccHalf green' );
    assert.equal( ff00ccHalf.b, 0xcc, 'ff00ccHalf blue' );
    assert.equal( ff00ccHalf.a, 0.5, 'ff00ccHalf alpha' );
    assert.equal( ff00ccHalf.toCSS(), 'rgba(255,0,204,0.5)', 'ff00ccHalf css' );
  } );

  QUnit.test( 'RGB/A direct', function( assert ) {
    var ff00cc = new Color( 0xff, 0x00, 0xcc );
    assert.equal( ff00cc.r, 0xff, 'ff00cc red' );
    assert.equal( ff00cc.g, 0, 'ff00cc green' );
    assert.equal( ff00cc.b, 0xcc, 'ff00cc blue' );
    assert.equal( ff00cc.a, 1, 'ff00cc alpha' );
    assert.equal( ff00cc.toCSS(), 'rgb(255,0,204)', 'ff00cc css' );

    var ff00ccHalf = new Color( 0xff, 0x00, 0xcc, 0.5 );
    assert.equal( ff00ccHalf.r, 0xff, 'ff00ccHalf red' );
    assert.equal( ff00ccHalf.g, 0, 'ff00ccHalf green' );
    assert.equal( ff00ccHalf.b, 0xcc, 'ff00ccHalf blue' );
    assert.equal( ff00ccHalf.a, 0.5, 'ff00ccHalf alpha' );
    assert.equal( ff00ccHalf.toCSS(), 'rgba(255,0,204,0.5)', 'ff00ccHalf css' );
  } );

  QUnit.test( 'Copy Constructor', function( assert ) {
    var ff00cc = new Color( 0xff, 0x00, 0xcc );
    var copy = new Color( ff00cc );

    assert.equal( ff00cc.r, copy.r );
    assert.equal( ff00cc.g, copy.g );
    assert.equal( ff00cc.b, copy.b );
    assert.equal( ff00cc.a, copy.a );
  } );

  QUnit.test( 'Keywords', function( assert ) {
    var yellow = new Color( 'yellow' );
    assert.equal( yellow.r, 0xff, 'yellow red' );
    assert.equal( yellow.g, 0xff, 'yellow green' );
    assert.equal( yellow.b, 0x00, 'yellow blue' );
    assert.equal( yellow.a, 1, 'yellow alpha' );

    var transparent = new Color( 'transparent' );
    assert.equal( transparent.r + transparent.g + transparent.b + transparent.a, 0, 'transparent sum' );
  } );

  QUnit.test( 'rgb', function( assert ) {
    var rgb = new Color( 'rgb(100,250,10)' );
    assert.equal( rgb.r, 100, 'rgb red' );
    assert.equal( rgb.g, 250, 'rgb green' );
    assert.equal( rgb.b, 10, 'rgb blue' );
    assert.equal( rgb.a, 1, 'rgb alpha' );
    assert.equal( rgb.toCSS(), 'rgb(100,250,10)', 'rgb css' );

    var clamped = new Color( 'rgb(-50,120%,999)' );
    assert.equal( clamped.r, 0, 'clamped rgb red' );
    assert.equal( clamped.g, 255, 'clamped rgb green' );
    assert.equal( clamped.b, 255, 'clamped rgb blue' );
  } );

  QUnit.test( 'rgba', function( assert ) {
    var rgba = new Color( 'rgba(100,100%,0%,0)' );
    assert.equal( rgba.r, 100, 'rgba red' );
    assert.equal( rgba.g, 255, 'rgba green' );
    assert.equal( rgba.b, 0, 'rgba blue' );
    assert.equal( rgba.a, 0, 'rgba alpha' );
    assert.equal( rgba.toCSS(), 'rgba(100,255,0,0)', 'rgba css' );

    var clamped = new Color( 'rgba(-50,120%,999,255)' );
    assert.equal( clamped.r, 0, 'clamped rgba red' );
    assert.equal( clamped.g, 255, 'clamped rgba green' );
    assert.equal( clamped.b, 255, 'clamped rgba blue' );
    assert.equal( clamped.a, 1, 'clamped rgba alpha' );
  } );

  QUnit.test( 'hsl', function( assert ) {
    var hsl = new Color( 'hsl(0,100%,50%)' );
    assert.equal( hsl.r, 255, 'hsl 1 red' );
    assert.equal( hsl.g, 0, 'hsl 1 green' );
    assert.equal( hsl.b, 0, 'hsl 1 blue' );

    hsl = new Color( 'hsl(0,0%,50%)' );
    assert.equal( hsl.r, 128, 'hsl 2 red' );
    assert.equal( hsl.g, 128, 'hsl 2 green' );
    assert.equal( hsl.b, 128, 'hsl 2 blue' );

    hsl = new Color( 'hsl(180,100%,50%)' );
    assert.equal( hsl.r, 0, 'hsl 3 red' );
    assert.equal( hsl.g, 255, 'hsl 3 green' );
    assert.equal( hsl.b, 255, 'hsl 3 blue' );

    hsl = new Color( 'hsl(90,25%,75%)' );
    assert.equal( hsl.r, 191, 'hsl 4 red' );
    assert.equal( hsl.g, 207, 'hsl 4 green' );
    assert.equal( hsl.b, 175, 'hsl 4 blue' );
  } );

  QUnit.test( 'hsla', function( assert ) {
    var hsl = new Color( 'hsla(90,25%,75%,0.25)' );
    assert.equal( hsl.r, 191, 'hsla red' );
    assert.equal( hsl.g, 207, 'hsla green' );
    assert.equal( hsl.b, 175, 'hsla blue' );
    assert.equal( hsl.a, 0.25, 'hsla alpha 0.25' );

    hsl = new Color( 'hsla(90,25%,75%,.25)' ); // without leading 0
    assert.equal( hsl.a, 0.25, 'hsla alpha .25' );
  } );
} );