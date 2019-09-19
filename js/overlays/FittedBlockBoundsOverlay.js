// Copyright 2015-2019, University of Colorado Boulder

/**
 * Shows the bounds of current fitted blocks.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const inherit = require( 'PHET_CORE/inherit' );
  const Matrix3 = require( 'DOT/Matrix3' );
  const Shape = require( 'KITE/Shape' );
  const ShapeBasedOverlay = require( 'SCENERY/overlays/ShapeBasedOverlay' );

  const scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );

  function FittedBlockBoundsOverlay( display, rootNode ) {
    ShapeBasedOverlay.call( this, display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  scenery.register( 'FittedBlockBoundsOverlay', FittedBlockBoundsOverlay );

  inherit( ShapeBasedOverlay, FittedBlockBoundsOverlay, {
    // @override
    addShapes: function() {
      const self = this;

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
          for ( let childDrawable = block.firstDrawable; childDrawable !== block.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
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
