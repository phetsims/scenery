// Copyright 2016, University of Colorado Boulder

/**
 * SVG drawable for Line nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var LineStatefulDrawable = require( 'SCENERY/display/drawables/LineStatefulDrawable' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGLineElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated SVGSelfDrawable whose purpose will be drawing our Line. One of these drawables will be created
   * for each displayed instance of a Line.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function LineSVGDrawable( renderer, instance ) {
    // Super-type initialization
    this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGLineElements ); // usesPaint: true

    // @protected {SVGLineElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
    this.svgElement = this.svgElement || document.createElementNS( scenery.svgns, 'line' );
  }

  scenery.register( 'LineSVGDrawable', LineSVGDrawable );

  inherit( SVGSelfDrawable, LineSVGDrawable, {
    /**
     * Updates the SVG elements so that they will appear like the current node's representation.
     * @protected
     *
     * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
     */
    updateSVGSelf: function() {
      var line = this.svgElement;

      if ( this.dirtyX1 ) {
        line.setAttribute( 'x1', this.node._x1 );
      }
      if ( this.dirtyY1 ) {
        line.setAttribute( 'y1', this.node._y1 );
      }
      if ( this.dirtyX2 ) {
        line.setAttribute( 'x2', this.node._x2 );
      }
      if ( this.dirtyY2 ) {
        line.setAttribute( 'y2', this.node._y2 );
      }

      // Apply any fill/stroke changes to our element.
      this.updateFillStrokeStyle( line );
    }
  } );
  LineStatefulDrawable.mixInto( LineSVGDrawable );

  Poolable.mixInto( LineSVGDrawable );

  return LineSVGDrawable;
} );
