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
  scenery.BackboneBlock = function BackboneBlock( instance, includeRoot ) {
    this.domElement = document.createElement( 'div' );
    this.includeRoot = includeRoot;
  };
  var BackboneBlock = scenery.BackboneBlock;
  
  inherit( Block, BackboneBlock, {
    getDOMDrawable: function() {
      return // TODO. maybe even have us conform to the API directly
    }
  } );
  
  return BackboneBlock;
} );
