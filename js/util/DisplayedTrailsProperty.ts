// Copyright 2022-2024, University of Colorado Boulder

/**
 * A Property that will contain a list of Trails where the root of the trail is a root Node of a Display, and the leaf
 * node is the provided Node.
 *
 * NOTE: If a Node is disposed, it will be removed from the trails.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import { Display, Node, scenery, Trail } from '../imports.js';
import optionize from '../../../phet-core/js/optionize.js';

type DisplayPredicate = Display | ( ( display: Display ) => boolean ) | null;

export type DisplayedTrailsPropertyOptions = {
  // If provided, we will only report trails that are rooted for the specific Display provided.
  display?: DisplayPredicate;

  // If true, we will follow the pdomParent if it is available (if our child node is specified in a pdomOrder of another
  // node, we will follow that order).
  // This essentially tracks the following:
  //
  // - followPdomOrder: true = visual trails (just children)
  // - followPdomORder: false = pdom trails (respecting pdomOrder)
  followPdomOrder?: boolean;

  // If true, we will only report trails where every node is visible: true.
  requireVisible?: boolean;

  // If true, we will only report trails where every node is pdomVisible: true.
  requirePdomVisible?: boolean;

  // If true, we will only report trails where every node is enabled: true.
  requireEnabled?: boolean;

  // If true, we will only report trails where every node is inputEnabled: true.
  requireInputEnabled?: boolean;

  // NOTE: Could think about adding pickability here in the future. The complication is that it doesn't measure our hit
  // testing precisely, because of pickable:null (default) and the potential existence of input listeners.
};

export default class DisplayedTrailsProperty extends TinyProperty<Trail[]> {

  public readonly node: Node;
  public readonly listenedNodeSet: Set<Node> = new Set<Node>();
  private readonly _trailUpdateListener: () => void;

  // Recorded options
  private readonly display: DisplayPredicate;
  private readonly followPdomOrder: boolean;
  private readonly requireVisible: boolean;
  private readonly requirePdomVisible: boolean;
  private readonly requireEnabled: boolean;
  private readonly requireInputEnabled: boolean;

  /**
   * We will contain Trails whose leaf node (lastNode) is this provided Node.
   */
  public constructor( node: Node, providedOptions?: DisplayedTrailsPropertyOptions ) {

    const options = optionize<DisplayedTrailsPropertyOptions>()( {
      // Listen to all displays
      display: null,

      // Default to visual trails (just children), with only pruning by normal visibility
      followPdomOrder: false,
      requireVisible: true,
      requirePdomVisible: false,
      requireEnabled: false,
      requireInputEnabled: false
    }, providedOptions );

    super( [] );

    // Save options for later updates
    this.node = node;
    this.display = options.display;
    this.followPdomOrder = options.followPdomOrder;
    this.requireVisible = options.requireVisible;
    this.requirePdomVisible = options.requirePdomVisible;
    this.requireEnabled = options.requireEnabled;
    this.requireInputEnabled = options.requireInputEnabled;

    this._trailUpdateListener = this.update.bind( this );

    this.update();
  }

  private update(): void {

    // Factored out because we're using a "function" below for recursion (NOT an arrow function)
    const display = this.display;
    const followPdomOrder = this.followPdomOrder;
    const requireVisible = this.requireVisible;
    const requirePdomVisible = this.requirePdomVisible;
    const requireEnabled = this.requireEnabled;
    const requireInputEnabled = this.requireInputEnabled;

    // Trails accumulated in our recursion that will be our Property's value
    const trails: Trail[] = [];

    // Nodes that were touched in the scan (we should listen to changes to ANY of these to see if there is a connection
    // or disconnection. This could potentially cause our Property to change
    const nodeSet = new Set<Node>();

    // Modified in-place during the search
    const trail = new Trail( this.node );

    // We will recursively add things to the "front" of the trail (ancestors)
    ( function recurse() {
      const root = trail.rootNode();

      // If a Node is disposed, we won't add listeners to it, so we abort slightly earlier.
      if ( root.isDisposed ) {
        return;
      }

      nodeSet.add( root );

      // If we fail other conditions, we won't add a trail OR recurse, but we will STILL have listeners added to the Node.
      if (
        ( requireVisible && !root.visible ) ||
        ( requirePdomVisible && !root.pdomVisible ) ||
        ( requireEnabled && !root.enabled ) ||
        ( requireInputEnabled && !root.inputEnabled )
      ) {
        return;
      }

      const displays = root.getRootedDisplays();

      let displayMatches: boolean;

      if ( display === null ) {
        displayMatches = displays.length > 0;
      }
      else if ( display instanceof Display ) {
        displayMatches = displays.includes( display );
      }
      else {
        displayMatches = displays.some( display );
      }

      if ( displayMatches ) {
        // Create a permanent copy that won't be mutated
        trails.push( trail.copy() );
      }

      const parents = followPdomOrder && root.pdomParent ? [ root.pdomParent ] : root.parents;

      parents.forEach( parent => {
        trail.addAncestor( parent );
        recurse();
        trail.removeAncestor();
      } );
    } )();

    // Add in new needed listeners
    nodeSet.forEach( node => {
      if ( !this.listenedNodeSet.has( node ) ) {
        this.addNodeListener( node );
      }
    } );

    // Remove listeners not needed anymore
    this.listenedNodeSet.forEach( node => {
      if ( !nodeSet.has( node ) ) {
        this.removeNodeListener( node );
      }
    } );

    // Guard in a way that deepEquality on the Property wouldn't (because of the Array wrapper)
    // NOTE: Duplicated with TrailsBetweenProperty, likely can be factored out.
    const currentTrails = this.value;
    let trailsEqual = currentTrails.length === trails.length;
    if ( trailsEqual ) {
      for ( let i = 0; i < trails.length; i++ ) {
        if ( !currentTrails[ i ].equals( trails[ i ] ) ) {
          trailsEqual = false;
          break;
        }
      }
    }

    if ( !trailsEqual ) {
      this.value = trails;
    }
  }

  private addNodeListener( node: Node ): void {
    this.listenedNodeSet.add( node );

    // Unconditional listeners, which affect all nodes.
    node.parentAddedEmitter.addListener( this._trailUpdateListener );
    node.parentRemovedEmitter.addListener( this._trailUpdateListener );
    node.rootedDisplayChangedEmitter.addListener( this._trailUpdateListener );
    node.disposeEmitter.addListener( this._trailUpdateListener );

    if ( this.followPdomOrder ) {
      node.pdomParentChangedEmitter.addListener( this._trailUpdateListener );
    }
    if ( this.requireVisible ) {
      node.visibleProperty.lazyLink( this._trailUpdateListener );
    }
    if ( this.requirePdomVisible ) {
      node.pdomVisibleProperty.lazyLink( this._trailUpdateListener );
    }
    if ( this.requireEnabled ) {
      node.enabledProperty.lazyLink( this._trailUpdateListener );
    }
    if ( this.requireInputEnabled ) {
      node.inputEnabledProperty.lazyLink( this._trailUpdateListener );
    }
  }

  private removeNodeListener( node: Node ): void {
    this.listenedNodeSet.delete( node );
    node.parentAddedEmitter.removeListener( this._trailUpdateListener );
    node.parentRemovedEmitter.removeListener( this._trailUpdateListener );
    node.rootedDisplayChangedEmitter.removeListener( this._trailUpdateListener );
    node.disposeEmitter.removeListener( this._trailUpdateListener );

    if ( this.followPdomOrder ) {
      node.pdomParentChangedEmitter.removeListener( this._trailUpdateListener );
    }
    if ( this.requireVisible ) {
      node.visibleProperty.unlink( this._trailUpdateListener );
    }
    if ( this.requirePdomVisible ) {
      node.pdomVisibleProperty.unlink( this._trailUpdateListener );
    }
    if ( this.requireEnabled ) {
      node.enabledProperty.unlink( this._trailUpdateListener );
    }
    if ( this.requireInputEnabled ) {
      node.inputEnabledProperty.unlink( this._trailUpdateListener );
    }
  }

  public override dispose(): void {
    this.listenedNodeSet.forEach( node => this.removeNodeListener( node ) );

    super.dispose();
  }
}

scenery.register( 'DisplayedTrailsProperty', DisplayedTrailsProperty );