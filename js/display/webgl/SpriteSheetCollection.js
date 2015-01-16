//  Copyright 2002-2014, University of Colorado Boulder

/**
 * Maintains a collection of SpriteSheets. Different regions are reserved for images based on a simple bin packing algorithm
 * if no space is left for a new Image on the currentSpriteSheet a new spriteSheet will be created on demand
 *
 * @author Sharfudeen Ashraf
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var SpriteSheet = require( 'SCENERY/display/webgl/SpriteSheet' );
  var FrameRange = require( 'SCENERY/display/webgl/FrameRange' );

  /**
   * @constructor
   */
  function SpriteSheetCollection() {
    this.spriteSheets = [ new SpriteSheet() ];
    this.imageFrameRangeCache = {};
  }

  return inherit( Object, SpriteSheetCollection, {
    /**
     * @param {Image} image
     * @returns {FrameRange}
     */
    addImage: function( image ) {

      //Check to see if any of the sprite sheets already contains that image
      if ( this.imageFrameRangeCache[ image.src ] ) {
        return this.imageFrameRangeCache[ image.src ];
      }
      var bounds = this.spriteSheets[ this.spriteSheets.length - 1 ].reserveImageSpace( image );
      if ( !bounds ) {
        var newSpriteSheet = new SpriteSheet();
        bounds = newSpriteSheet.reserveImageSpace( image );
        this.spriteSheets.push( newSpriteSheet );
      }
      var frameRange = new FrameRange( bounds, this.spriteSheets.length - 1 );
      this.imageFrameRangeCache[ image.src ] = frameRange;
      return frameRange;
    }

  } );

} );
