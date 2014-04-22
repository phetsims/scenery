// Copyright 2002-2014, University of Colorado Boulder

/**
 * VBox arranges the child nodes vertically, and they can be centered, left or right justified.
 * Vertical spacing can be set as a constant or a function which depends on the adjacent nodes.
 *
 * See a dynamic test in scenery\tests\test-vbox.html
 *
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var AbstractBox = require( 'SCENERY/nodes/AbstractBox' );

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
  scenery.HBox = function VBox( options ) {
    AbstractBox.call( this, 'horizontal', function() {
      var minY = _.min( _.map( this.children, function( child ) {return child.top;} ) );
      var maxY = _.max( _.map( this.children, function( child ) {return child.top + child.height;} ) );
      var centerY = (maxY + minY) / 2;

      //Start at x=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {x:number} option.
      var x = 0;
      for ( var i = 0; i < this.children.length; i++ ) {
        var child = this.children[i];
        child.left = x;

        //Set the position horizontally
        if ( this.options.align === 'top' ) {
          child.top = minY;
        }
        else if ( this.options.align === 'bottom' ) {
          child.bottom = maxY;
        }
        else {//default to center
          child.centerY = centerY;
        }

        //Move to the next vertical position.
        x += child.width + this.options.spacing( child, this.children[i + 1] );
      }
    }, options );
  };

  return inherit( AbstractBox, scenery.HBox );
} );
