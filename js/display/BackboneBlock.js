// Copyright 2002-2014, University of Colorado Boulder

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

  scenery.BackboneBlock = function BackboneBlock() {
    this._domElement = document.createElement( 'div' );
  };
  var BackboneBlock = scenery.BackboneBlock;

  inherit( Block, BackboneBlock, {
    getDomElement: function() {
      return this._domElement;
    }
  } );

  return BackboneBlock;
} );
