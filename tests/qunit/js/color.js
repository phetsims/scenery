
(function(){
  'use strict';
  
  module( 'Scenery: Color' );
  
  var Color = scenery.Color;
  
  test( 'RGB Hex', function() {
    var ff00cc = new Color( '#ff00cc' );
    equal( ff00cc.r, 0xff, 'ff00cc red' );
    equal( ff00cc.g, 0, 'ff00cc green' );
    equal( ff00cc.b, 0xcc, 'ff00cc blue' );
    equal( ff00cc.a, 1, 'ff00cc alpha' );
    equal( ff00cc.getCSS(), 'rgb(255,0,204)', 'ff00cc css' );
    
    var f0c = new Color( '#f0c' );
    equal( f0c.r, 0xff, 'f0c red' );
    equal( f0c.g, 0, 'f0c green' );
    equal( f0c.b, 0xcc, 'f0c blue' );
    equal( f0c.a, 1, 'f0c alpha' );
    equal( f0c.getCSS(), 'rgb(255,0,204)', 'f0c css' );
  } );
  
  test( 'Keywords', function() {
    var yellow = new Color( 'yellow' );
    equal( yellow.r, 0xff, 'yellow red' );
    equal( yellow.g, 0xff, 'yellow green' );
    equal( yellow.b, 0x00, 'yellow blue' );
    equal( yellow.a, 1, 'yellow alpha' );
    
    var transparent = new Color( 'transparent' );
    equal( transparent.r + transparent.g + transparent.b + transparent.a, 0, 'transparent sum' );
  } );
  
  test( 'rgb', function() {
    var rgb = new Color( 'rgb(100,250,10)' );
    equal( rgb.r, 100, 'rgb red' );
    equal( rgb.g, 250, 'rgb green' );
    equal( rgb.b, 10, 'rgb blue' );
    equal( rgb.a, 1, 'rgb alpha' );
    equal( rgb.getCSS(), 'rgb(100,250,10)', 'rgb css' );
    
    var clamped = new Color( 'rgb(-50,120%,999)' );
    equal( clamped.r, 0, 'clamped rgb red' );
    equal( clamped.g, 255, 'clamped rgb green' );
    equal( clamped.b, 255, 'clamped rgb blue' );
  } );
  
  test( 'rgba', function() {
    var rgba = new Color( 'rgba(100,100%,0%,0)' );
    equal( rgba.r, 100, 'rgba red' );
    equal( rgba.g, 255, 'rgba green' );
    equal( rgba.b, 0, 'rgba blue' );
    equal( rgba.a, 0, 'rgba alpha' );
    equal( rgba.getCSS(), 'rgba(100,255,0,0)', 'rgba css' );
    
    var clamped = new Color( 'rgba(-50,120%,999,255)' );
    equal( clamped.r, 0, 'clamped rgba red' );
    equal( clamped.g, 255, 'clamped rgba green' );
    equal( clamped.b, 255, 'clamped rgba blue' );
    equal( clamped.a, 1, 'clamped rgba alpha' );
  } );
  
  test( 'hsl', function() {
    var hsl = new Color( 'hsl(0,100%,50%)' );
    equal( hsl.r, 255, 'hsl 1 red' );
    equal( hsl.g, 0, 'hsl 1 green' );
    equal( hsl.b, 0, 'hsl 1 blue' );
    
    hsl = new Color( 'hsl(0,0%,50%)' );
    equal( hsl.r, 128, 'hsl 2 red' );
    equal( hsl.g, 128, 'hsl 2 green' );
    equal( hsl.b, 128, 'hsl 2 blue' );
    
    hsl = new Color( 'hsl(180,100%,50%)' );
    equal( hsl.r, 0, 'hsl 3 red' );
    equal( hsl.g, 255, 'hsl 3 green' );
    equal( hsl.b, 255, 'hsl 3 blue' );
    
    hsl = new Color( 'hsl(90,25%,75%)' );
    equal( hsl.r, 191, 'hsl 4 red' );
    equal( hsl.g, 207, 'hsl 4 green' );
    equal( hsl.b, 175, 'hsl 4 blue' );
  } );
  
  test( 'hsla', function() {
    var hsl = new Color( 'hsla(90,25%,75%,0.25)' );
    equal( hsl.r, 191, 'hsla red' );
    equal( hsl.g, 207, 'hsla green' );
    equal( hsl.b, 175, 'hsla blue' );
    equal( hsl.a, 0.25, 'hsla alpha 0.25' );
    
    hsl = new Color( 'hsla(90,25%,75%,.25)' ); // without leading 0
    equal( hsl.a, 0.25, 'hsla alpha .25' );
  } );
})();
