// Copyright 2024-2025, University of Colorado Boulder

/**
 * A Property that will contain a list of Trails where the root of the trail is a root Node of a Display, and the leaf
 * node is the provided Node.
 *
 * // REVIEW: This is a very complicated component and deserves a bit more doc. Some ideas about what to explain:
 * // REVIEW:   1. That this is synchronously updated and doesn't listen to instances.
 * // REVIEW:   2.
 * // REVIEW:   2.
 * // REVIEW:   2.
 *
 * // REVIEW: can you describe this a bit more. Do you mean any Node in a trail? What about if the provided Node is disposed?
 * NOTE: If a Node is disposed, it will be removed from the trails.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import optionize from '../../../phet-core/js/optionize.js';
import type Display from '../display/Display.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Trail from '../util/Trail.js';

type DisplayPredicate = Display | ( ( display: Display ) => boolean ) | null;

export type DisplayedTrailsPropertyOptions = {
  // If provided, we will only report trails that are rooted for the specific Display provided.
  display?: DisplayPredicate;

  // If true, we will additionally follow the pdomParent if it is available (if our child node is specified in a pdomOrder of another
  // node, we will follow that order).
  // This essentially tracks the following:
  //
  // REVIEW: I'd actually add [a-z]?Pdom[A-Z] to phet/bad-sim-text if you're alright with that. Close to https://github.com/phetsims/chipper/blob/f56c273970f22f857bc8f5bd0148f256534a702f/eslint/rules/bad-sim-text.js#L35-L36
  //
  // REVIEW: Aren't these boolean values opposite? followPDOMOrder:true should respect pdomOrder. Also, it isn't clear
  //         from the doc how you ask for "all trails, visual or PDOM". Is that part of the featureset? I believe
  //         that likely we would always force visible as a base feature, and only add on visibility, but this should
  //         be explained. As easy as the doc update above I just did: "we will _additionally_ follow the pdomParent"
  // - followPDOMOrder: true = visual trails (just children)
  // - followPDOMOrder: false = pdom trails (respecting pdomOrder)
  followPDOMOrder?: boolean;

  // If true, we will only report trails where every node is visible: true.
  requireVisible?: boolean;

  // If true, we will only report trails where every node is pdomVisible: true.
  requirePDOMVisible?: boolean;

  // If true, we will only report trails where every node is enabled: true.
  requireEnabled?: boolean;

  // If true, we will only report trails where every node is inputEnabled: true.
  requireInputEnabled?: boolean;

  // REVIEW: Instead of following the same feature above, can we just use `pickable:false` to help us prune. I agree
  //           it may not be worth while to listen to the combo of pickable+inputListenerLength. Can you describe what benefit
  //           we may get by adding in Pickable listening?
  // NOTE: Could think about adding pickability here in the future. The complication is that it doesn't measure our hit
  // testing precisely, because of pickable:null (default) and the potential existence of input listeners.
};

export default class DisplayedTrailsProperty extends TinyProperty<Trail[]> {

  // REVIEW: How about a rename like "targetNode", no strong preference if you don't want to.
  public readonly node: Node;

  // REVIEW: Please add doc why we only need to listen to a Node once, even if it is in multiple trails?
  public readonly listenedNodeSet: Set<Node> = new Set<Node>();
  private readonly _trailUpdateListener: () => void;

  // Recorded options
  // REVIEW: Please rename this and the option to something less confusing. Perhaps `displaySupport`, or
  // `whichDisplay`, or something that sounds like it could be a predicate.
  private readonly display: DisplayPredicate;
  private readonly followPDOMOrder: boolean;
  private readonly requireVisible: boolean;
  private readonly requirePDOMVisible: boolean;
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
      followPDOMOrder: false,
      requireVisible: true,
      requirePDOMVisible: false,
      requireEnabled: false,
      requireInputEnabled: false
    }, providedOptions );

    super( [] );

    // Save options for later updates
    this.node = node;
    this.display = options.display;
    this.followPDOMOrder = options.followPDOMOrder;
    this.requireVisible = options.requireVisible;
    this.requirePDOMVisible = options.requirePDOMVisible;
    this.requireEnabled = options.requireEnabled;
    this.requireInputEnabled = options.requireInputEnabled;

    this._trailUpdateListener = this.update.bind( this );

    this.update();
  }

  private update(): void {

    // Factored out because we're using a "function" below for recursion (NOT an arrow function)
    const display = this.display;
    const followPDOMOrder = this.followPDOMOrder;
    const requireVisible = this.requireVisible;
    const requirePDOMVisible = this.requirePDOMVisible;
    const requireEnabled = this.requireEnabled;
    const requireInputEnabled = this.requireInputEnabled;

    // Trails accumulated in our recursion that will be our Property's value
    const trails: Trail[] = [];

    // Nodes that were touched in the scan (we should listen to changes to ANY of these to see if there is a connection
    // or disconnection). This could potentially cause our Property to change
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

      // REVIEW: Please say why we need listeners on this Node. Also please confirm (via doc) that adding
      // If we fail other conditions, we won't add a trail OR recurse, but we will STILL have listeners added to the Node.
      if (
        ( requireVisible && !root.visible ) ||
        ( requirePDOMVisible && !root.pdomVisible ) ||
        ( requireEnabled && !root.enabled ) ||
        ( requireInputEnabled && !root.inputEnabled )
      ) {
        return;
      }

      const displays = root.getRootedDisplays();

      // REVIEW: initialize to false?
      let displayMatches: boolean;

      if ( display === null ) {
        displayMatches = displays.length > 0;
      }
      else if ( typeof display !== 'function' ) {
        displayMatches = displays.includes( display );
      }
      else {
        displayMatches = displays.some( display );
      }

      if ( displayMatches ) {
        // Create a permanent copy that won't be mutated
        trails.push( trail.copy() );
      }

      // REVIEW: I'm officially confused about this feature. What is the value of "either or", why not be able to
      // support both visual and PDOM in the same Property? If this is indeed best, please be sure to explain where
      // the option is defined.
      const parents = followPDOMOrder && root.pdomParent ? [ root.pdomParent ] : root.parents;

      parents.forEach( parent => {
        trail.addAncestor( parent );
        recurse();
        trail.removeAncestor();
      } );
    } )();

    // REVIEW: Webstorm flagged the next 29 lines as duplicated with TrailsBetweenProperty. Let's factor that our or fix that somehow.
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
    // REVIEW: ^^^^ +1, yes please factor out.
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

    // REVIEW: Can this be improved upon by utilizing a custom valueComparisonStrategy? I don't see that being much
    // less performant given that you are doing all the above work on each call to update().
    if ( !trailsEqual ) {
      this.value = trails;
    }
  }

  // REVIEW: Rename to either `addNodeListeners`, or something more general like `listenToNode()`.
  private addNodeListener( node: Node ): void {
    this.listenedNodeSet.add( node );

    // Unconditional listeners, which affect all nodes.
    node.parentAddedEmitter.addListener( this._trailUpdateListener );
    node.parentRemovedEmitter.addListener( this._trailUpdateListener );
    node.rootedDisplayChangedEmitter.addListener( this._trailUpdateListener );
    node.disposeEmitter.addListener( this._trailUpdateListener );

    if ( this.followPDOMOrder ) {
      node.pdomParentChangedEmitter.addListener( this._trailUpdateListener );
    }
    if ( this.requireVisible ) {
      node.visibleProperty.lazyLink( this._trailUpdateListener );
    }
    if ( this.requirePDOMVisible ) {
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

    if ( this.followPDOMOrder ) {
      node.pdomParentChangedEmitter.removeListener( this._trailUpdateListener );
    }
    if ( this.requireVisible ) {
      node.visibleProperty.unlink( this._trailUpdateListener );
    }
    if ( this.requirePDOMVisible ) {
      node.pdomVisibleProperty.unlink( this._trailUpdateListener );
    }
    if ( this.requireEnabled ) {
      node.enabledProperty.unlink( this._trailUpdateListener );
    }
    if ( this.requireInputEnabled ) {
      node.inputEnabledProperty.unlink( this._trailUpdateListener );
    }
  }

  // REVIEW: I always forget why you don't need to also clear your reference to the provided Node. Do you?
  // REVIEW: Also maybe assert here that your provided node is in this listened to Node set?
  public override dispose(): void {
    this.listenedNodeSet.forEach( node => this.removeNodeListener( node ) );

    super.dispose();
  }
}

scenery.register( 'DisplayedTrailsProperty', DisplayedTrailsProperty );