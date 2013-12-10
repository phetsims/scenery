// Copyright 2002-2013, University of Colorado

/**
 * A "backbone" block that controls a DOM element (usually a div) that contains other blocks with DOM/SVG/Canvas/WebGL content
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Block = require( 'SCENERY/display/Block' );
  
  // includeRoot is used for the root of a display, where the instance should be thought of as fully "under" the backbone
  scenery.BackboneBlock = function BackboneBlock( instance, renderer, includeRoot, existingDiv ) {
    this.instance = instance;
    this.renderer = renderer;
    this.domElement = existingDiv || BackboneBlock.createDivBackbone();
    this.includeRoot = includeRoot;
    
    // TODO: flesh out into stitch handling:
    var drawable = instance.firstDrawable;
    while ( drawable ) {
      this.domElement.appendChild( drawable.domElement );
      if ( drawable === instance.lastDrawable ) {
        break;
      }
      drawable = drawable.nextDrawable;
    }
  };
  var BackboneBlock = scenery.BackboneBlock;
  
  inherit( Block, BackboneBlock, {
    getDOMDrawable: function() {
      return new scenery.DOMElementDrawable( this.instance.trail, this.renderer, this.domElement );
    }
  } );
  
  BackboneBlock.createDivBackbone = function() {
    var div = document.createElement( 'div' );
    div.style.position = 'absolute';
    div.style.left = '0';
    div.style.top = '0';
    div.style.width = '0';
    div.style.height = '0';
    return div;
  };
  
  return BackboneBlock;
} );
