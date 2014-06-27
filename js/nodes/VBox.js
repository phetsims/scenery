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
   * align:   How to line up the items horizontally.  One of 'center', 'left' or 'right'.  Defaults to 'center'.
   *
   * @constructor
   */
  scenery.VBox = function VBox( options ) {
    AbstractBox.call( this, 'vertical', function() {
      var minX = _.min( _.map( this.children, function( child ) {return child.left;} ) );
      var maxX = _.max( _.map( this.children, function( child ) {return child.left + child.width;} ) );
      var centerX = (maxX + minX) / 2;

      //Start at y=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {y:number} option.
      var y = 0;
      for ( var i = 0; i < this.children.length; i++ ) {
        var child = this.children[i];
        child.top = y;

        //Set the position horizontally
        if ( this.options.align === 'left' ) {
          child.left = minX;
        }
        else if ( this.options.align === 'right' ) {
          child.right = maxX;
        }
        else {//default to center
          child.centerX = centerX;
        }

        //Move to the next vertical position.
        y += child.height + this.options.spacing( child, this.children[i + 1] );
      }
    }, options );
  };

  return inherit( AbstractBox, scenery.VBox );
} );
