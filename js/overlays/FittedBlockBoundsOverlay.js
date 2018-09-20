// Copyright 2015, University of Colorado Boulder

/**
 * Shows the bounds of current fitted blocks.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Shape = require( 'KITE/Shape' );
  var ShapeBasedOverlay = require( 'SCENERY/overlays/ShapeBasedOverlay' );

  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );

  function FittedBlockBoundsOverlay( display, rootNode ) {
    ShapeBasedOverlay.call( this, display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  scenery.register( 'FittedBlockBoundsOverlay', FittedBlockBoundsOverlay );

  inherit( ShapeBasedOverlay, FittedBlockBoundsOverlay, {
    // @override
    addShapes: function() {
      var self = this;

      function processBackbone( backbone, matrix ) {
        if ( backbone.willApplyTransform ) {
          matrix = matrix.timesMatrix( backbone.backboneInstance.relativeTransform.matrix );
        }
        backbone.blocks.forEach( function( block ) {
          processBlock( block, matrix );
        } );
      }

      function processBlock( block, matrix ) {
        if ( block.fitBounds && !block.fitBounds.isEmpty() ) {
          self.addShape( Shape.bounds( block.fitBounds ).transformed( matrix ), 'rgba(255,0,0,0.8)', true );
        }
        if ( block.firstDrawable && block.lastDrawable ) {
          for ( var childDrawable = block.firstDrawable; childDrawable !== block.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
            processDrawable( childDrawable, matrix );
          }
          processDrawable( block.lastDrawable, matrix );
        }
      }

      function processDrawable( drawable, matrix ) {
        // How we detect backbones (for now)
        if ( drawable.backboneInstance ) {
          processBackbone( drawable, matrix );
        }
      }

      processBackbone( this.display._rootBackbone, Matrix3.IDENTITY );
    }
  } );

  return FittedBlockBoundsOverlay;
} );
