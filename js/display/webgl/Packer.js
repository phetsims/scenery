// Copyright 2002-2014, University of Colorado Boulder

/**
 * Modified version of https://github.com/jakesgordon/bin-packing/blob/master/js/packer.js
 *
 * @author Sharfudeen Ashraf
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var BlockItem = require( 'SCENERY/display/webgl/BlockItem' );
  var Vector2 = require( 'DOT/Vector2' );


  /**
   * @param {number} width // overall width of the spriteSheet
   * @param {number} height
   * @constructor
   */
  function Packer( width, height ) {
    this.root = new BlockItem( 0, 0, width, height );
  }
  return inherit( Object, Packer, {

    /**
     * returns the starting position (x,y) within SpriteSheet where s space for a  given width, height can be reserved or
     * returns 'null' if the given width, height  cannot be accommodated.
     * @param {number} width
     * @param {number} height
     */
    reserveSpace: function( width, height ) {
      var position = null;
      var node = this.findBlock( this.root, width, height );
      if ( node ) {
        position = new Vector2( node.x, node.y );

        //Mark the block as "used" and split the remaining space into two available portions (right and down)
        this.splitBlock( node, width, height );
      }
      return position;
    },

    /**
     * @private
     * @param parent
     * @param {number} w
     * @param {number} h
     * @returns {*}
     */
    findBlock: function( parent, w, h ) {
      if ( parent.used ) {
        return this.findBlock( parent.right, w, h ) || this.findBlock( parent.down, w, h );
      }
      else if ( (w <= parent.width) && (h <= parent.height) ) {
        return parent;
      }
      else {
        return null;
      }
    },

    /**
     * @private
     * @param {BlockItem} block
     * @param w
     * @param h
     */
    splitBlock: function( block, w, h ) {
      block.used = true;
      //the dimension of the space available to the right of the "node"
      block.right = new BlockItem( block.x + w, block.y, block.width - w, h );
      block.down = new BlockItem( block.x, block.y + h, block.width, block.height - h );
    }

  } );

} );
