// Copyright 2002-2014, University of Colorado Boulder


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
  var scenery = require( 'SCENERY/scenery' );
  var Renderer = require( 'SCENERY/display/Renderer' );

  var summaryBits = [
    // renderer bits (part of a normal renderer bitmask)
    Renderer.bitmaskCanvas,
    Renderer.bitmaskSVG,
    Renderer.bitmaskDOM,
    Renderer.bitmaskWebGL,
    Renderer.bitmaskPixi,

    // summary bits (added to the renderer bitmask to handle special flags for the summary)
    Renderer.bitmaskSingleCanvas,
    Renderer.bitmaskSingleSVG,
    Renderer.bitmaskNotPainted,
    Renderer.bitmaskBoundsValid
  ];
  var numSummaryBits = summaryBits.length;

  // A bitmask with all of the bits set that we record
  var bitmaskAll = 0;
  for ( var l = 0; l < numSummaryBits; l++ ) {
    bitmaskAll |= summaryBits[ l ];
  }

  scenery.RendererSummary = function RendererSummary( node ) {
    // NOTE: assumes that we are created in the Node constructor
    assert && assert( node._rendererBitmask === Renderer.bitmaskNodeDefault, 'Node must have a default bitmask when creating a RendererSummary' );
    assert && assert( node._children.length === 0, 'Node cannot have children when creating a RendererSummary' );

    this.node = node;

    // Maps stringified bitmask bit (e.g. "1" for Canvas, since Renderer.bitmaskCanvas is 0x01) to
    // a count of how many children (or self) have that property (e.g. can't renderer all of their contents with Canvas)
    this._counts = {};
    for ( var i = 0; i < numSummaryBits; i++ ) {
      this._counts[ summaryBits[ i ] ] = 0; // set everything to 0 at first
    }

    // @public
    this.bitmask = bitmaskAll;

    this.selfBitmask = RendererSummary.summaryBitmaskForNodeSelf( node );

    this.summaryChange( this.bitmask, this.selfBitmask );

    // required listeners to update our summary based on painted/non-painted information
    var listener = this.selfChange.bind( this );
    this.node.on( 'opacity', listener );
    this.node.on( 'hint', listener ); // should fire on things like node.renderer being changed
    this.node.on( 'clip', listener );
    this.node.on( 'selfBoundsValid', listener ); // e.g. Text, may change based on boundsMethod
  };
  var RendererSummary = scenery.RendererSummary;

  inherit( Object, RendererSummary, {
    /*
     * @public
     * Use a bitmask of all 1s to represent 'does not exist' since we count zeros
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

    isRendererSupported: function( renderer ) {
      return !!( renderer & this.bitmask );
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

    areBoundsValid: function() {
      return !!( Renderer.bitmaskBoundsValid & this.bitmask );
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

      // NOTE: If changing, see Instance.updateRenderingState
      var requiresSplit = node._hints.requireElement || node._hints.cssTransform || node._hints.layerSplit;
      var mightUseOpacity = node.opacity !== 1 || node._hints.usesOpacity;
      var mightUseClip = node._clipArea !== null;
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
           !mightUseOpacity && // Opacity not supported for Canvas blocks yet
           !mightUseClip && // Clipping not supported for Canvas blocks yet
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

      return bitmask;
    },

    // for debugging purposes
    bitToString: function( bit ) {
      if ( bit === Renderer.bitmaskCanvas ) { return 'C'; }
      if ( bit === Renderer.bitmaskSVG ) { return 'S'; }
      if ( bit === Renderer.bitmaskDOM ) { return 'D'; }
      if ( bit === Renderer.bitmaskWebGL ) { return 'W'; }
      if ( bit === Renderer.bitmaskPixi ) { return 'P'; }
      if ( bit === Renderer.bitmaskSingleCanvas ) { return 'c'; }
      if ( bit === Renderer.bitmaskSingleSVG ) { return 's'; }
      if ( bit === Renderer.bitmaskNotPainted ) { return 'n'; }
      if ( bit === Renderer.bitmaskBoundsValid ) { return 'b'; }
      return '?';
    },

    // for debugging purposes
    bitmaskToString: function( bitmask ) {
      var result = '';
      for ( var i = 0; i < numSummaryBits; i++ ) {
        var bit = summaryBits[ i ];
        if ( bitmask & bit ) {
          result += RendererSummary.bitToString( bit );
        }
      }
      return result;
    }
  } );

  return RendererSummary;
} );
