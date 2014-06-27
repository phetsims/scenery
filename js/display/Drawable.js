// Copyright 2002-2014, University of Colorado

/**
 * A unit that is drawable with a specific renderer.
 * NOTE: Drawables are assumed to be pooled with Poolable, as freeToPool() is called
 *
 * APIs for drawable types:
 *
 * DOM: {
 *   domElement: {HTMLElement}
 * }
 *
 * OHTWO TODO: add more API information, and update
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var globalId = 1;

  scenery.Drawable = function Drawable( renderer ) {
    this.initializeDrawable( renderer );
  };
  var Drawable = scenery.Drawable;

  inherit( Object, Drawable, {
    initializeDrawable: function( renderer ) {
      assert && assert( !this.id || this.disposed, 'If we previously existed, we need to have been disposed' );

      // unique ID for drawables
      this.id = this.id || globalId++;

      sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( '[' + this.constructor.name + '*] initialize ' + this.toString() );

      this.cleanDrawable();

      this.renderer = renderer;

      this.dirty = true;
      this.disposed = false;

      this.linksDirty = false;

      return this;
    },

    cleanDrawable: function() {
      // what drawble we are being rendered (or put) into (will be filled in later)
      this.parentDrawable = null;
      this.backbone = null; // a backbone reference (if applicable).

      this.pendingParentDrawable = null; // what our parent drawable will be after the stitch is finished
      this.pendingBackbone = null;       // what our backbone will be after the stitch is finished (if applicable)
      this.pendingAddition = false;      // whether we are to be added to a block/backbone in our updateBlock() call
      this.pendingRemoval = false;       // whether we are to be removed from a block/backbone in our updateBlock() call

      assert && assert( !this.previousDrawable && !this.nextDrawable,
        'By cleaning (disposal or fresh creation), we should have disconnected from the linked list' );

      // linked list handling (will be filled in later)
      this.previousDrawable = null;
      this.nextDrawable = null;

      // similar but without recent changes, so that we can traverse both orders at the same time for stitching
      this.oldPreviousDrawable = null;
      this.oldNextDrawable = null;
    },

    // called to add a block (us) as a child of a backbone
    setBlockBackbone: function( backboneInstance ) {
      sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( '[' + this.constructor.name + '*] setBlockBackbone ' +
                                                                this.toString() + ' with ' + backboneInstance.toString() );

      // if this is being called, Block will be guaranteed to be loaded
      assert && assert( this instanceof scenery.Block );

      this.parentDrawable = backboneInstance;
      this.backbone = backboneInstance;
      this.pendingParentDrawable = backboneInstance;
      this.pendingBackbone = backboneInstance;
      this.pendingAddition = false;
      this.pendingRemoval = false;
    },

    notePendingAddition: function( display, block, backbone ) {
      sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( '[' + this.constructor.name + '*] notePendingAddition ' +
                                                                this.toString() + ' with ' + block.toString() + ', ' +
                                                                ( backbone ? backbone.toString() : '-' ) );

      assert && assert( backbone !== undefined, 'backbone can be either null or a backbone' );
      assert && assert( block instanceof scenery.Block );

      this.pendingParentDrawable = block;
      this.pendingBackbone = backbone;
      this.pendingAddition = true;

      // if we weren't already marked for an update, mark us
      if ( !this.pendingRemoval ) {
        display.markDrawableChangedBlock( this );
      }
    },

    notePendingRemoval: function( display ) {
      sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( '[' + this.constructor.name + '*] notePendingRemoval ' +
                                                                this.toString() );

      this.pendingRemoval = true;

      // if we weren't already marked for an update, mark us
      if ( !this.pendingAddition ) {
        display.markDrawableChangedBlock( this );
      }
    },

    // moving a drawable that isn't changing backbones, just potentially changing its block.
    // it should not have notePendingAddition or notePendingRemoval called on it.
    notePendingMove: function( display, block ) {
      sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( '[' + this.constructor.name + '*] notePendingMove ' +
                                                                this.toString() + ' with ' + block.toString() );

      assert && assert( block instanceof scenery.Block );

      this.pendingParentDrawable = block;

      if ( !this.pendingRemoval || !this.pendingAddition ) {
        display.markDrawableChangedBlock( this );
      }

      // set both flags, since we need it to be removed and added
      this.pendingAddition = true;
      this.pendingRemoval = true;
    },

    updateBlock: function() {
      sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( '[' + this.constructor.name + '*] updateBlock ' + this.toString() +
                                                                ' with add:' + this.pendingAddition +
                                                                ' remove:' + this.pendingRemoval +
                                                                ' old:' + ( this.parentDrawable ? this.parentDrawable.toString() : '-' ) +
                                                                ' new:' + ( this.pendingParentDrawable ? this.pendingParentDrawable.toString() : '-' ) );
      sceneryLog && sceneryLog.Drawable && sceneryLog.push();

      if ( this.pendingRemoval || this.pendingAddition ) {
        // we are only unchanged if we have an addition AND removal, and the endpoints are identical
        var changed = !this.pendingRemoval || !this.pendingAddition ||
                      this.parentDrawable !== this.pendingParentDrawable ||
                      this.backbone !== this.pendingBackbone;

        if ( changed ) {
          if ( this.pendingRemoval ) {
            sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( 'removing from ' + this.parentDrawable.toString() );
            this.parentDrawable.removeDrawable( this );
          }

          this.parentDrawable = this.pendingParentDrawable;
          this.backbone = this.pendingBackbone;

          if ( this.pendingAddition ) {
            sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( 'adding to ' + this.parentDrawable.toString() );
            this.parentDrawable.addDrawable( this );
          }
        }
        else {
          sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( 'unchanged' );
        }

        this.pendingAddition = false;
        this.pendingRemoval = false;
      }

      sceneryLog && sceneryLog.Drawable && sceneryLog.pop();
    },

    updateLinks: function() {
      this.oldNextDrawable = this.nextDrawable;
      this.oldPreviousDrawable = this.previousDrawable;
      this.linksDirty = false;
    },

    markDirty: function() {
      if ( !this.dirty ) {
        this.dirty = true;

        // TODO: notify what we want to call repaint() later
        if ( this.parentDrawable ) {
          this.parentDrawable.markDirtyDrawable( this );
        }
      }
    },

    // will ensure that after syncTree phase is done, we will have updateLinks() called on us
    markLinksDirty: function( display ) {
      if ( !this.linksDirty ) {
        this.linksDirty = true;
        display.markDrawableForLinksUpdate( this );
      }
    },

    // marks us for disposal in the next phase of updateDisplay(), and disconnects from the linked list
    markForDisposal: function( display ) {
      // as we are marked for disposal, we disconnect from the linked list (so our disposal setting nulls won't cause issues)
      Drawable.disconnectBefore( this, display );
      Drawable.disconnectAfter( this, display );

      display.markDrawableForDisposal( this );
    },

    // disposes immediately, and makes no guarantees about out linked list's state (disconnects).
    disposeImmediately: function( display ) {
      // as we are marked for disposal, we disconnect from the linked list (so our disposal setting nulls won't cause issues)
      Drawable.disconnectBefore( this, display );
      Drawable.disconnectAfter( this, display );

      this.dispose();
    },

    // generally do not call this directly, use markForDisposal (so Display will dispose us), or disposeImmediately.
    dispose: function() {
      assert && assert( !this.disposed, 'We should not re-dispose drawables' );

      sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( '[' + this.constructor.name + '*] dispose ' + this.toString() );

      this.cleanDrawable();
      this.disposed = true;

      // for now
      this.freeToPool();
    },

    audit: function( allowPendingBlock, allowPendingList, allowDirty ) {
      if ( assertSlow ) {
        assertSlow && assertSlow( !this.disposed,
          'If we are being audited, we assume we are in the drawable display tree, and we should not be marked as disposed' );
        assertSlow && assertSlow( this.renderer, 'Should not have a 0 (no) renderer' );

        assertSlow && assertSlow( !this.backbone || this.parentDrawable,
          'If we have a backbone reference, we must have a parentDrawable (our block)' );

        if ( !allowPendingBlock ) {
          assertSlow && assertSlow( !this.pendingAddition );
          assertSlow && assertSlow( !this.pendingRemoval );
          assertSlow && assertSlow( this.parentDrawable === this.pendingParentDrawable,
            'Assure our parent and pending parent match, if we have updated blocks' );
          assertSlow && assertSlow( this.backbone === this.pendingBackbone,
            'Assure our backbone and pending backbone match, if we have updated blocks' );
        }

        if ( !allowPendingList ) {
          assertSlow && assertSlow( this.oldPreviousDrawable === this.previousDrawable,
            'Pending linked-list references should be cleared by now' );
          assertSlow && assertSlow( this.oldNextDrawable === this.nextDrawable,
            'Pending linked-list references should be cleared by now' );
          assertSlow && assertSlow( !this.linksDirty, 'Links dirty flag should be clean' );
        }

        if ( !allowDirty ) {
          assertSlow && assertSlow( !this.dirty,
            'Should not be dirty at this phase, if we are in the drawable display tree' );
        }
      }
    },

    toString: function() {
      return this.constructor.name + '#' + this.id;
    },

    toDetailedString: function() {
      return this.toString();
    }
  } );

  // a,b {Drawable}, connects the two drawables in the linked list, while cutting the previous connection and marking
  // the links for updates.
  Drawable.connectDrawables = function( a, b, display ) {
    // we don't need to do anything if there is no change
    if ( a.nextDrawable !== b ) {
      // touch previous neighbors
      if ( a.nextDrawable ) {
        a.nextDrawable.markLinksDirty( display );
        a.nextDrawable.previousDrawable = null;
      }
      if ( b.previousDrawable ) {
        b.previousDrawable.markLinksDirty( display );
        b.previousDrawable.nextDrawable = null;
      }

      a.nextDrawable = b;
      b.previousDrawable = a;

      // mark these as needing updates
      a.markLinksDirty( display );
      b.markLinksDirty( display );
    }
  };

  Drawable.disconnectBefore = function( a, display ) {
    // we don't need to do anything if there is no change
    if ( a.previousDrawable ) {
      a.markLinksDirty( display );
      a.previousDrawable.markLinksDirty( display );
      a.previousDrawable.nextDrawable = null;
      a.previousDrawable = null;
    }
  };

  Drawable.disconnectAfter = function( a, display ) {
    // we don't need to do anything if there is no change
    if ( a.nextDrawable ) {
      a.markLinksDirty( display );
      a.nextDrawable.markLinksDirty( display );
      a.nextDrawable.previousDrawable = null;
      a.nextDrawable = null;
    }
  };

  // converts a linked list of drawables to an array (useful for debugging/assertion purposes, should not be used in production code)
  Drawable.listToArray = function( firstDrawable, lastDrawable ) {
    var arr = [];

    // assumes we'll hit lastDrawable, otherwise we'll NPE
    for ( var drawable = firstDrawable; ; drawable = drawable.nextDrawable ) {
      arr.push( drawable );

      if ( drawable === lastDrawable ) {
        break;
      }
    }

    return arr;
  };

  // converts an old linked list of drawables to an array (useful for debugging/assertion purposes, should not be used in production code)
  Drawable.oldListToArray = function( firstDrawable, lastDrawable ) {
    var arr = [];

    // assumes we'll hit lastDrawable, otherwise we'll NPE
    for ( var drawable = firstDrawable; ; drawable = drawable.oldNextDrawable ) {
      arr.push( drawable );

      if ( drawable === lastDrawable ) {
        break;
      }
    }

    return arr;
  };

  return Drawable;
} );
