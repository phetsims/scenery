// Copyright 2013-2016, University of Colorado Boulder

/**
 * Contains information about what renderers (and a few other flags) are supported for an entire subtree.
 *
 * We effectively do this by tracking bitmask changes from scenery.js (used for rendering properties in general). In particular, we count
 * how many zeros in the bitmask we have in key places.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );

  var summaryBits = [
    // renderer bits ("Is renderer X supported by the entire sub-tree?")
    Renderer.bitmaskCanvas,
    Renderer.bitmaskSVG,
    Renderer.bitmaskDOM,
    Renderer.bitmaskWebGL,

    // summary bits (added to the renderer bitmask to handle special flags for the summary)
    Renderer.bitmaskSingleCanvas,
    Renderer.bitmaskSingleSVG,
    Renderer.bitmaskNotPainted,
    Renderer.bitmaskBoundsValid,
    Renderer.bitmaskNotAccessible, // NOTE: This could be separated out into its own implementation for this flag, since
    // there are cases where we actually have nothing accessible DUE to things being pulled out by another order.
    // This is generally NOT the case, so I've left this in here because it significantly simplifies the implementation.

    // inverse renderer bits ("Do all painted nodes NOT support renderer X in this sub-tree?")
    Renderer.bitmaskLacksCanvas,
    Renderer.bitmaskLacksSVG,
    Renderer.bitmaskLacksDOM,
    Renderer.bitmaskLacksWebGL
  ];
  var numSummaryBits = summaryBits.length;

  // A bitmask with all of the bits set that we record
  var bitmaskAll = 0;
  for ( var l = 0; l < numSummaryBits; l++ ) {
    bitmaskAll |= summaryBits[ l ];
  }

  /**
   * @constructor
   *
   * @param {Node} node
   */
  function RendererSummary( node ) {
    assert && assert( node instanceof scenery.Node );

    // NOTE: assumes that we are created in the Node constructor
    assert && assert( node._rendererBitmask === Renderer.bitmaskNodeDefault, 'Node must have a default bitmask when creating a RendererSummary' );
    assert && assert( node._children.length === 0, 'Node cannot have children when creating a RendererSummary' );

    // @private {Node}
    this.node = node;

    // @private {Object} Maps stringified bitmask bit (e.g. "1" for Canvas, since Renderer.bitmaskCanvas is 0x01) to
    // a count of how many children (or self) have that property (e.g. can't renderer all of their contents with Canvas)
    this._counts = {};
    for ( var i = 0; i < numSummaryBits; i++ ) {
      this._counts[ summaryBits[ i ] ] = 0; // set everything to 0 at first
    }

    // @public {number} (scenery-internal)
    this.bitmask = bitmaskAll;

    // @private {number}
    this.selfBitmask = RendererSummary.summaryBitmaskForNodeSelf( node );

    this.summaryChange( this.bitmask, this.selfBitmask );

    // required listeners to update our summary based on painted/non-painted information
    var listener = this.selfChange.bind( this );
    this.node.onStatic( 'opacity', listener );
    this.node.onStatic( 'hint', listener ); // should fire on things like node.renderer being changed
    this.node.onStatic( 'clip', listener );
    this.node.onStatic( 'selfBoundsValid', listener ); // e.g. Text, may change based on boundsMethod
    this.node.onStatic( 'accessibleContent', listener );
    this.node.onStatic( 'accessibleOrder', listener );
  }

  scenery.register( 'RendererSummary', RendererSummary );

  inherit( Object, RendererSummary, {
    /**
     * Use a bitmask of all 1s to represent 'does not exist' since we count zeros
     * @public
     *
     * @param {number} oldBitmask
     * @param {number} newBitmask
     */
    summaryChange: function( oldBitmask, newBitmask ) {
      assert && this.audit();

      var changeBitmask = oldBitmask ^ newBitmask; // bit set only if it changed

      var ancestorOldMask = 0;
      var ancestorNewMask = 0;
      for ( var i = 0; i < numSummaryBits; i++ ) {
        var bit = summaryBits[ i ];

        // If the bit for the renderer has changed
        if ( bit & changeBitmask ) {

          // If it is now set (wasn't before), gained support for the renderer
          if ( bit & newBitmask ) {
            this._counts[ bit ]--; // reduce count, since we count the number of 0s (unsupported)
            if ( this._counts[ bit ] === 0 ) {
              ancestorNewMask |= bit; // add our bit to the "new" mask we will send to ancestors
            }
          }
          // It was set before (now isn't), lost support for the renderer
          else {
            this._counts[ bit ]++; // increment the count, since we count the number of 0s (unsupported)
            if ( this._counts[ bit ] === 1 ) {
              ancestorOldMask |= bit; // add our bit to the "old" mask we will send to ancestors
            }
          }
        }
      }

      if ( ancestorOldMask || ancestorNewMask ) {

        var oldSubtreeBitmask = this.bitmask;
        assert && assert( oldSubtreeBitmask !== undefined );

        for ( var j = 0; j < numSummaryBits; j++ ) {
          var ancestorBit = summaryBits[ j ];
          // Check for added bits
          if ( ancestorNewMask & ancestorBit ) {
            this.bitmask |= ancestorBit;
          }

          // Check for removed bits
          if ( ancestorOldMask & ancestorBit ) {
            this.bitmask ^= ancestorBit;
            assert && assert( !( this.bitmask & ancestorBit ),
              'Should be cleared, doing cheaper XOR assuming it already was set' );
          }
        }

        this.node.trigger0( 'rendererSummary' ); // please don't change children when listening to this!
        this.node.onSummaryChange( oldSubtreeBitmask, this.bitmask );

        var len = this.node._parents.length;
        for ( var k = 0; k < len; k++ ) {
          this.node._parents[ k ]._rendererSummary.summaryChange( ancestorOldMask, ancestorNewMask );
        }

        assert && assert( this.bitmask === this.computeBitmask(), 'Sanity check' );
      }

      assert && this.audit();
    },

    // @public
    selfChange: function() {
      var oldBitmask = this.selfBitmask;
      var newBitmask = RendererSummary.summaryBitmaskForNodeSelf( this.node );
      if ( oldBitmask !== newBitmask ) {
        this.summaryChange( oldBitmask, newBitmask );
        this.selfBitmask = newBitmask;
      }
    },

    // @private
    computeBitmask: function() {
      var bitmask = 0;
      for ( var i = 0; i < numSummaryBits; i++ ) {
        if ( this._counts[ summaryBits[ i ] ] === 0 ) {
          bitmask |= summaryBits[ i ];
        }
      }
      return bitmask;
    },

    /**
     * @public
     * Is the renderer compatible with every single painted node under this subtree?
     * (Can this entire sub-tree be rendered with just this renderer)
     *
     * @param {number} renderer - Single bit preferred. If multiple bits set, requires ALL painted nodes are compatible
     *                            with ALL of the bits.
     */
    isSubtreeFullyCompatible: function( renderer ) {
      return !!( renderer & this.bitmask );
    },

    /**
     * @public
     * Is the renderer compatible with at least one painted node under this subtree?
     *
     * @param {number} renderer - Single bit preferred. If multiple bits set, will return if a single painted node is
     *                            compatible with at least one of the bits.
     */
    isSubtreeContainingCompatible: function( renderer ) {
      return !( ( renderer << Renderer.bitmaskLacksShift ) & this.bitmask );
    },

    isSingleCanvasSupported: function() {
      return !!( Renderer.bitmaskSingleCanvas & this.bitmask );
    },

    isSingleSVGSupported: function() {
      return !!( Renderer.bitmaskSingleSVG & this.bitmask );
    },

    isNotPainted: function() {
      return !!( Renderer.bitmaskNotPainted & this.bitmask );
    },

    isNotAccessible: function() {
      return !!( Renderer.bitmaskNotAccessible & this.bitmask );
    },

    areBoundsValid: function() {
      return !!( Renderer.bitmaskBoundsValid & this.bitmask );
    },

    /**
     * Given a bitmask representing a list of ordered preferred renderers, we check to see if all of our nodes can be
     * displayed in a single SVG block, AND that given the preferred renderers, that it will actually happen in our
     * rendering process.
     */
    isSubtreeRenderedExclusivelySVG: function( preferredRenderers ) {
      // Check if we have anything that would PREVENT us from having a single SVG block
      if ( !this.isSingleSVGSupported() ) {
        return false;
      }

      // Check for any renderer preferences that would CAUSE us to choose not to display with a single SVG block
      for ( var i = 0; i < Renderer.numActiveRenderers; i++ ) {
        // Grab the next-most preferred renderer
        var renderer = Renderer.bitmaskOrder( preferredRenderers, i );

        // If it's SVG, congrats! Everything will render in SVG (since SVG is supported, as noted above)
        if ( Renderer.bitmaskSVG & renderer ) {
          return true;
        }

        // Since it's not SVG, if there's a single painted node that supports this renderer (which is preferred over SVG),
        // then it will be rendered with this renderer, NOT SVG.
        if ( this.isSubtreeContainingCompatible( renderer ) ) {
          return false;
        }
      }

      return false; // sanity check
    },

    /**
     * Given a bitmask representing a list of ordered preferred renderers, we check to see if all of our nodes can be
     * displayed in a single Canvas block, AND that given the preferred renderers, that it will actually happen in our
     * rendering process.
     */
    isSubtreeRenderedExclusivelyCanvas: function( preferredRenderers ) {
      // Check if we have anything that would PREVENT us from having a single Canvas block
      if ( !this.isSingleCanvasSupported() ) {
        return false;
      }

      // Check for any renderer preferences that would CAUSE us to choose not to display with a single Canvas block
      for ( var i = 0; i < Renderer.numActiveRenderers; i++ ) {
        // Grab the next-most preferred renderer
        var renderer = Renderer.bitmaskOrder( preferredRenderers, i );

        // If it's Canvas, congrats! Everything will render in Canvas (since Canvas is supported, as noted above)
        if ( Renderer.bitmaskCanvas & renderer ) {
          return true;
        }

        // Since it's not Canvas, if there's a single painted node that supports this renderer (which is preferred over Canvas),
        // then it will be rendered with this renderer, NOT Canvas.
        if ( this.isSubtreeContainingCompatible( renderer ) ) {
          return false;
        }
      }

      return false; // sanity check
    },

    // for debugging purposes
    audit: function() {
      if ( assert ) {
        for ( var i = 0; i < numSummaryBits; i++ ) {
          var bit = summaryBits[ i ];
          var countIsZero = this._counts[ bit ] === 0;
          var bitmaskContainsBit = !!( this.bitmask & bit );
          assert( countIsZero === bitmaskContainsBit, 'Bits should be set if count is zero' );
        }
      }
    },

    // for debugging purposes
    toString: function() {
      var result = RendererSummary.bitmaskToString( this.bitmask );
      for ( var i = 0; i < numSummaryBits; i++ ) {
        var bit = summaryBits[ i ];
        var countForBit = this._counts[ bit ];
        if ( countForBit !== 0 ) {
          result += ' ' + RendererSummary.bitToString( bit ) + ':' + countForBit;
        }
      }
      return result;
    }
  }, {
    bitmaskAll: bitmaskAll,

    /**
     * Determines which of the summary bits can be set for a specific Node (ignoring children/ancestors).
     * For instance, for bitmaskSingleSVG, we only don't include the flag if THIS node prevents its usage
     * (even though child nodes may prevent it in the renderer summary itself).
     *
     * @param {Node} node
     */
    summaryBitmaskForNodeSelf: function( node ) {
      var bitmask = node._rendererBitmask;

      if ( node.isPainted() ) {
        bitmask |= ( ( node._rendererBitmask & Renderer.bitmaskCurrentRendererArea ) ^ Renderer.bitmaskCurrentRendererArea ) << Renderer.bitmaskLacksShift;
      }
      else {
        bitmask |= Renderer.bitmaskCurrentRendererArea << Renderer.bitmaskLacksShift;
      }

      // NOTE: If changing, see Instance.updateRenderingState
      var requiresSplit = node._hints.requireElement || node._hints.cssTransform || node._hints.layerSplit;
      var rendererHint = node._hints.renderer;

      // Whether this subtree will be able to support a single SVG element
      // NOTE: If changing, see Instance.updateRenderingState
      if ( !requiresSplit && // Can't have a single SVG element if we are split
           Renderer.isSVG( node._rendererBitmask ) && // If our node doesn't support SVG, can't do it
           ( !rendererHint || Renderer.isSVG( rendererHint ) ) ) { // Can't if a renderer hint is set to something else
        bitmask |= Renderer.bitmaskSingleSVG;
      }

      // Whether this subtree will be able to support a single Canvas element
      // NOTE: If changing, see Instance.updateRenderingState
      if ( !requiresSplit && // Can't have a single SVG element if we are split
           Renderer.isCanvas( node._rendererBitmask ) && // If our node doesn't support Canvas, can't do it
           ( !rendererHint || Renderer.isCanvas( rendererHint ) ) ) { // Can't if a renderer hint is set to something else
        bitmask |= Renderer.bitmaskSingleCanvas;
      }

      if ( !node.isPainted() ) {
        bitmask |= Renderer.bitmaskNotPainted;
      }
      if ( node.areSelfBoundsValid() ) {
        bitmask |= Renderer.bitmaskBoundsValid;
      }
      if ( !node.accessibleContent && !node.hasAccessibleOrder() ) {
        bitmask |= Renderer.bitmaskNotAccessible;
      }

      return bitmask;
    },

    // for debugging purposes
    bitToString: function( bit ) {
      if ( bit === Renderer.bitmaskCanvas ) { return 'Canvas'; }
      if ( bit === Renderer.bitmaskSVG ) { return 'SVG'; }
      if ( bit === Renderer.bitmaskDOM ) { return 'DOM'; }
      if ( bit === Renderer.bitmaskWebGL ) { return 'WebGL'; }
      if ( bit === Renderer.bitmaskLacksCanvas ) { return '(-Canvas)'; }
      if ( bit === Renderer.bitmaskLacksSVG ) { return '(-SVG)'; }
      if ( bit === Renderer.bitmaskLacksDOM ) { return '(-DOM)'; }
      if ( bit === Renderer.bitmaskLacksWebGL ) { return '(-WebGL)'; }
      if ( bit === Renderer.bitmaskSingleCanvas ) { return 'SingleCanvas'; }
      if ( bit === Renderer.bitmaskSingleSVG ) { return 'SingleSVG'; }
      if ( bit === Renderer.bitmaskNotPainted ) { return 'NotPainted'; }
      if ( bit === Renderer.bitmaskBoundsValid ) { return 'BoundsValid'; }
      if ( bit === Renderer.bitmaskNotAccessible ) { return 'NotAccessible'; }
      return '?';
    },

    // for debugging purposes
    bitmaskToString: function( bitmask ) {
      var result = '';
      for ( var i = 0; i < numSummaryBits; i++ ) {
        var bit = summaryBits[ i ];
        if ( bitmask & bit ) {
          result += RendererSummary.bitToString( bit ) + ' ';
        }
      }
      return result;
    }
  } );

  return RendererSummary;
} );
