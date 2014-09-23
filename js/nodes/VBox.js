// Copyright 2002-2013, University of Colorado Boulder

/**
 * VBox arranges the child nodes vertically, and they can be centered, left or right justified.
 * Vertical spacing can be set as a constant or a function which depends on the adjacent nodes.
 *
 * @deprecated, please use LayoutBox instead, see https://github.com/phetsims/scenery/issues/281
 *
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var LayoutBox = require( 'SCENERY/nodes/LayoutBox' );

  /**
   *
   * @param options Same as Node.constructor.options with the following additions:
   *
   * spacing: can be a number or a function.  If a number, then it will be the vertical spacing between each object.
   *              If a function, then the function will have the signature function(top,bottom){} which returns the spacing between adjacent pairs of items.
   * align:   How to line up the items horizontally.  One of 'center', 'top' or 'bottom'.  Defaults to 'center'.
   *
   * @constructor
   */
  function VBox( options ) {
    assert && assert( !options.hasOwnProperty( 'orientation' ) );
    options.orientation = 'vertical';
    LayoutBox.call( this, options );
  }

  scenery.VBox = VBox;

  return inherit( LayoutBox, VBox );
} );